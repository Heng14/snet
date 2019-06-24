import os
from skimage import io
import tensorflow as tf
from utils import *
import utils
import keras.backend as K
import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator
from snet import snet
from grad_cam import *
import time

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range = 0.1,
    horizontal_flip=True,
    vertical_flip=True)

class SurvivalModel:

	def __init__(self):

		self.model_builder = snet


	def fit(self, datasets_train, datasets_val, datasets_test, train_name=None, val_name=None, test_name=None, loss_func='hinge', epochs=500, lr=0.001, mode='merge', batch_size = 8):

		self.datasets_train = datasets_train
		self.datasets_val = datasets_val
		self.datasets_test = datasets_test
		self.loss_func = loss_func
		self.epochs = epochs
		self.lr = lr
		self.batch_size = batch_size
		self.train_name = train_name
		self.val_name = val_name
		self.test_name = test_name
		self.__build_graph()
		
		## train the model
		if mode=='merge':
		  self.__train_merge()
		elif mode=='infer':
		  self.__infer()
		elif mode=='vis':
		  self.__vis()
		elif mode=='vis_cam':
		  self.__vis_cam()


	def __build_graph(self):

		input_shape = self.datasets_train[0][0].shape[1:]

		with tf.name_scope('input'):
			X = tf.placeholder(dtype=tf.float32, shape=(None, )+input_shape, name='X')
			time_value = tf.placeholder(dtype=tf.float32, shape=(None, ), name='time_value')
			event = tf.placeholder(dtype=tf.int16, shape=(None, ), name='event')

		with tf.name_scope('model'):
			#self.model = self.model_builder(input_shape[0], input_shape[0])
			#self.model = self.model_builder((11, 224, 224), 1)
			self.model = self.model_builder()

		with tf.name_scope('output'):
			score = tf.identity(self.model(X), name='score')

		with tf.name_scope('metric'):
			ci = self.__concordance_index(score, time_value, event)
			if self.loss_func=='hinge':
				loss = self.__hinge_loss(score, time_value, event)
			elif self.loss_func=='log':
				loss = self.__log_loss(score, time_value, event)
			elif self.loss_func=='cox':
				loss = self.__cox_loss(score, time_value, event)

		with tf.name_scope('train'):

			num_epoch = tf.Variable(0, name='global_step', trainable=False)
			boundaries = [5, 25, 70]
			#learing_rates = [0.0001, 0.0005, 0.00001, 0.00005]
			learing_rates = [0.0001, 0.0001, 0.0001, 0.0001]
			learing_rate = tf.train.piecewise_constant(num_epoch, boundaries=boundaries, values=learing_rates)
			optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate)
			train_op = optimizer.minimize(loss, name='train_op')

		## save the tensors and ops so that we can use them later
		self.__X = X
		self.__time_value = time_value
		self.__event = event
		self.__score = score
		self.__ci = ci
		self.__loss = loss
		self.__train_op = train_op	
			
	def __infer(self):
		weights_path = "my_model_weights_35.h5"
		self.__sess = tf.Session()
		K.set_session(self.__sess)
		self.__sess.run(tf.global_variables_initializer())
		self.model.load_weights(weights_path, by_name=True)
		self.__print_loss_ci_infer(self.datasets_train, self.train_name, 'train')
		self.__print_loss_ci_infer(self.datasets_val, self.val_name, 'val')
		self.__print_loss_ci_infer(self.datasets_test, self.test_name, 'test')

	def __vis(self):
		weights_path = "checkpoints/my_model_weights_35.h5"
		self.__sess = tf.Session()
		K.set_session(self.__sess)
		self.__sess.run(tf.global_variables_initializer())
		self.model.load_weights(weights_path, by_name=True)
		self.__vis_layer(self.datasets_train, self.train_name, 'train')

	def __vis_cam(self):
		weights_path = "checkpoints/my_model_weights_113.h5"
		#self.__sess = tf.Session()
		#K.set_session(self.__sess)
		#self.__sess.run(tf.global_variables_initializer())
		self.model.load_weights(weights_path, by_name=True)
		#self.__vis_layer_cam(self.datasets_train, self.train_name, 'train')
		#self.__vis_layer_cam(self.datasets_val, self.val_name, 'val')
		self.__vis_layer_cam(self.datasets_test, self.test_name, 'test')


	def __train_merge(self):

		## Merge training datasets
		X_train, time_value_train, event_train = utils.combine_datasets(self.datasets_train)

		## start training
		self.__sess = tf.Session()
		K.set_session(self.__sess)
		self.__sess.run(tf.global_variables_initializer())
		#self.model.load_weights(weights_path, by_name=True)

		file_train = open(f"train_log.txt","w")
		file_val = open(f"val_log.txt","w")
		file_test = open(f"test_log.txt","w")

		print (f'pre epoch train log:')
		ci_train, loss_train = self.__print_loss_ci(self.datasets_train)
		ci_val, loss_val = self.__print_loss_ci(self.datasets_val)
		ci_test, loss_test = self.__print_loss_ci(self.datasets_test)
		
		file_train.write(str(ci_train)+" "+str(loss_train)+"\n")
		file_val.write(str(ci_val)+" "+str(loss_val)+"\n")
		file_test.write(str(ci_test)+" "+str(loss_test)+"\n")

		ci = 0

		for epoch in range(self.epochs):

			#next_batch, num_batches = utils.batch_factory(X_train, time_value_train, event_train, self.batch_size)
			#for _ in range(num_batches):
			#	X_batch, time_value_batch, event_batch = next_batch()


			time_value_event = np.hstack((time_value_train[...,None], event_train[...,None]))			
			batches = 0
			for X_batch, Y_batch in datagen.flow(X_train, time_value_event, batch_size=self.batch_size):
				batches += 1
				if batches >= X_train.shape[0] // self.batch_size:
					break
				time_value_batch = Y_batch[:,0]
				event_batch = Y_batch[:,1]
			
				self.__sess.run(self.__train_op, feed_dict={self.__X: X_batch, self.__time_value: time_value_batch, self.__event: event_batch, K.learning_phase(): 1})

			print (f'epoch {epoch} train log:')
			ci_train, loss_train = self.__print_loss_ci(self.datasets_train)
			file_train.write(str(ci_train)+" "+str(loss_train)+"\n")

			print('-'*20 + 'Epoch: {0}'.format(epoch) + '-'*20)
			print (f'epoch {epoch} val log:')
			ci_val, loss_val = self.__print_loss_ci(self.datasets_val)
			file_val.write(str(ci_val)+" "+str(loss_val)+"\n")
			#if val_ci > ci:
			#	ci = val_ci
			#	self.model.save_weights(f'my_model_weights_{epoch}.h5')
			#	print ("model saved!")
			print (f'epoch {epoch} test log:')
			ci_test, loss_test = self.__print_loss_ci(self.datasets_test)
			file_test.write(str(ci_test)+" "+str(loss_test)+"\n")
			if ci_test > ci:
				ci = ci_test
				#self.model.save_weights(f'my_model_weights_{epoch}.h5')
				print ("model saved!")

			if epoch>500:
				file_train.close()
				file_val.close()
				file_test.close()
				raise


	def predict(self, X_test):
		'''
		Args
			X: design matrix of shape (num_samples, ) + input_shape
		'''

		assert X_test.shape[1:]==self.datasets_train[0][0].shape[1:], 'Shapes of testing and training data must equal'
		
		return self.__sess.run(self.__score, feed_dict = {self.__X: X_test, K.learning_phase():0})



	def evaluate(self, X_test, time_value_test, event_test):
		'''
		Evaluate the loss and c-index of the model for the given test data
		'''
		assert X_test.shape[1:]==self.datasets_train[0][0].shape[1:], 'Shapes of testing and training data must equal'

		return self.__sess.run([self.__loss, self.__ci], feed_dict = {self.__X: X_test, self.__time_value: time_value_test, self.__event: event_test, K.learning_phase(): 1})



	def __concordance_index(self, score, time_value, event):
		'''
		Args
			score: 		predicted score, tf tensor of shape (None, )
			time_value:		true survival time_value, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		'''

		## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
		ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1)<time_value, tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

		## count how many score[i]<score[j]
		s1 = tf.gather(score, ix[:,0])
		s2 = tf.gather(score, ix[:,1])
		ci = tf.reduce_mean(tf.cast(s1<s2, tf.float32), name='c_index')

		return ci



	def __hinge_loss(self, score, time_value, event):
		'''
		Args
			score:	 	predicted score, tf tensor of shape (None, 1)
			time_value:		true survival time_value, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		'''

		## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
		ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1)<time_value, tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

		## if score[i]>score[j], incur hinge loss
		s1 = tf.gather(score, ix[:,0])
		s2 = tf.gather(score, ix[:,1])
		loss = tf.reduce_mean(tf.maximum(1+s1-s2, 0.0), name='loss')

		return loss



	def __log_loss(self, score, time_value, event):
		'''
		Args
			score: 	predicted survival time_value, tf tensor of shape (None, 1)
			time_value:		true survival time_value, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		'''

		## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
		ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1)<time_value, tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

		## if score[i]>score[j], incur log loss
		s1 = tf.gather(score, ix[:,0])
		s2 = tf.gather(score, ix[:,1])
		loss = tf.reduce_mean(tf.log(1+tf.exp(s1-s2)), name='loss')

		return loss



	def __cox_loss(self, score, time_value, event):
		'''
		Args
			score: 		predicted survival time_value, tf tensor of shape (None, 1)
			time_value:		true survival time_value, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		Return
			loss:		partial likelihood of cox regression
		'''


		## cox regression computes the risk score, we want the opposite
		score = -score

		## find index i satisfying event[i]==1
		ix = tf.where(tf.cast(event, tf.bool)) # shape of ix is [None, 1]

		## sel_mat is a matrix where sel_mat[i,j]==1 where time_value[i]<=time_value[j]
		sel_mat = tf.cast(tf.gather(time_value, ix)<=time_value, tf.float32)

		## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time_value[i]<=time_value[j] and event[i]==1
		p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))
		#p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(tf.transpose(tf.exp(score)), axis=-1))
		loss = -tf.reduce_mean(p_lik)

		return loss


	def __print_loss_ci(self, datasets):
		'''
		This compute loss and ci on the whole dataset. It is reasonable.
		'''
		## losses and c-indices on traning
		#loss_train = np.zeros(len(self.datasets_train))
		#ci_train = np.zeros(len(self.datasets_train))

		X, time_value, event = zip(*datasets)
		X = np.concatenate(X, axis=0)
		time_value = np.concatenate(time_value, axis=0)
		event = np.concatenate(event, axis=0)
		assert len(X) == len(time_value) == len(event), print ('X, time_value, event len are not equal!!!')

		loss_sum = 0
		ci_sum = 0
		count = 0
		bs = 16

		for i in range(len(X)//bs):
			X_batch, time_value_batch, event_batch = X[bs*i:bs*(i+1),...], time_value[bs*i:bs*(i+1),...], event[bs*i:bs*(i+1),...]

			start = time.time()
			score_i, time_value_i, event_i = self.__sess.run([self.__score, self.__time_value, self.__event], feed_dict = {self.__X: X_batch, self.__time_value: time_value_batch, self.__event: event_batch, K.learning_phase(): 0})
			end = time.time()
			#print ("time:", end-start)

			if i == 0:
				socre_all = score_i
				time_value_all = time_value_i
				event_all = event_i
			else:
				socre_all = np.concatenate((socre_all, score_i), axis=0)
				time_value_all = np.concatenate((time_value_all, time_value_i), axis=0)
				event_all = np.concatenate((event_all, event_i), axis=0)
				
			#loss_i, ci_i = self.evaluate(X_batch, time_value_batch, event_batch)
			#print ('loss_i, ci_i: ',loss_i, ci_i)

			#if math.isnan(loss_i) or math.isnan(ci_i) or math.isinf(loss_i) or math.isinf(ci_i):
			#	print ('here')
			#	continue
			#loss_sum += loss_i*bs
			#ci_sum += ci_i*bs
			#count += bs             

			#print (loss_i, ci_i, math.isnan(loss_i), math.isnan(ci_i))
			#print (X_batch.shape, time_value_batch.shape, event_batch.shape)
		#print (loss_sum, ci_sum, count)

		i += 1
		X_batch, time_value_batch, event_batch = X[bs*i:len(X),...], time_value[bs*i:len(time_value),...], event[bs*i:len(event),...]
		#loss_i, ci_i = self.evaluate(X_batch, time_value_batch, event_batch)

		score_i, time_value_i, event_i = self.__sess.run([self.__score, self.__time_value, self.__event], feed_dict = {self.__X: X_batch, self.__time_value: time_value_batch, self.__event: event_batch, K.learning_phase(): 0})

		socre_all = np.concatenate((socre_all, score_i), axis=0)
		time_value_all = np.concatenate((time_value_all, time_value_i), axis=0)
		event_all = np.concatenate((event_all, event_i), axis=0)
		#print (socre_all.shape, time_value_all.shape, event_all.shape)

		'''
		ci = self.__concordance_index(socre_all, time_value_all, event_all)
		if self.loss_func=='hinge':
			loss = self.__hinge_loss(socre_all, time_value_all, event_all)
		elif self.loss_func=='log':
			loss = self.__log_loss(socre_all, time_value_all, event_all)
		elif self.loss_func=='cox':
			loss = self.__cox_loss(socre_all, time_value_all, event_all)		
		'''


		loss, ci = self.__sess.run([self.__loss, self.__ci], feed_dict={self.__score: socre_all, self.__time_value: time_value_all, self.__event: event_all, K.learning_phase(): 0})

		#print (loss, ci)
		#raise

		#if not (math.isnan(loss_i) or math.isnan(ci_i) or math.isinf(loss_i) or math.isinf(ci_i)):
		#	loss_sum += loss_i*(len(X)-bs*i)
		#	ci_sum += ci_i*(len(X)-bs*i)	
		#	count += (len(X)-bs*i)


		## print them
		#print('loss={0}'.format(round(loss, 3)))
		#print('ci={0}'.format(round(ci, 3)))
		print(f'loss={loss}')
		print(f'ci={ci}')
		print()
		return ci, loss

	def __print_loss_ci_infer(self, datasets, dataset_name, flag = 'train'):
		'''
		This compute loss and ci on the whole dataset. It is reasonable.
		'''
		## losses and c-indices on traning
		#loss_train = np.zeros(len(self.datasets_train))
		#ci_train = np.zeros(len(self.datasets_train))

		X, time_value, event = zip(*datasets)
		X = np.concatenate(X, axis=0)
		time_value = np.concatenate(time_value, axis=0)
		event = np.concatenate(event, axis=0)
		assert len(X) == len(time_value) == len(event), print ('X, time_value, event len are not equal!!!')

		loss_sum = 0
		ci_sum = 0
		count = 0
		bs = 16

		for i in range(len(X)//bs):
			X_batch, time_value_batch, event_batch = X[bs*i:bs*(i+1),...], time_value[bs*i:bs*(i+1),...], event[bs*i:bs*(i+1),...]

			score_i, time_value_i, event_i = self.__sess.run([self.__score, self.__time_value, self.__event], feed_dict = {self.__X: X_batch, self.__time_value: time_value_batch, self.__event: event_batch, K.learning_phase(): 0})

			if i == 0:
				socre_all = score_i
				time_value_all = time_value_i
				event_all = event_i
			else:
				socre_all = np.concatenate((socre_all, score_i), axis=0)
				time_value_all = np.concatenate((time_value_all, time_value_i), axis=0)
				event_all = np.concatenate((event_all, event_i), axis=0)
				
			#loss_i, ci_i = self.evaluate(X_batch, time_value_batch, event_batch)
			#print ('loss_i, ci_i: ',loss_i, ci_i)

			#if math.isnan(loss_i) or math.isnan(ci_i) or math.isinf(loss_i) or math.isinf(ci_i):
			#	print ('here')
			#	continue
			#loss_sum += loss_i*bs
			#ci_sum += ci_i*bs
			#count += bs             

			#print (loss_i, ci_i, math.isnan(loss_i), math.isnan(ci_i))
			#print (X_batch.shape, time_value_batch.shape, event_batch.shape)
		#print (loss_sum, ci_sum, count)

		i += 1
		X_batch, time_value_batch, event_batch = X[bs*i:len(X),...], time_value[bs*i:len(time_value),...], event[bs*i:len(event),...]
		#loss_i, ci_i = self.evaluate(X_batch, time_value_batch, event_batch)

		score_i, time_value_i, event_i = self.__sess.run([self.__score, self.__time_value, self.__event], feed_dict = {self.__X: X_batch, self.__time_value: time_value_batch, self.__event: event_batch, K.learning_phase(): 0})

		socre_all = np.concatenate((socre_all, score_i), axis=0)
		time_value_all = np.concatenate((time_value_all, time_value_i), axis=0)
		event_all = np.concatenate((event_all, event_i), axis=0)
		#print (socre_all.shape, time_value_all.shape, event_all.shape)

		#print (len(socre_all), len(time_value_all), len(event_all))
		cal_ci(socre_all, time_value_all, event_all, dataset_name, flag)


		'''
		ci = self.__concordance_index(socre_all, time_value_all, event_all)
		if self.loss_func=='hinge':
			loss = self.__hinge_loss(socre_all, time_value_all, event_all)
		elif self.loss_func=='log':
			loss = self.__log_loss(socre_all, time_value_all, event_all)
		elif self.loss_func=='cox':
			loss = self.__cox_loss(socre_all, time_value_all, event_all)		
		'''


		loss, ci = self.__sess.run([self.__loss, self.__ci], feed_dict={self.__score: socre_all, self.__time_value: time_value_all, self.__event: event_all, K.learning_phase(): 0})

		#print (loss, ci)
		#raise

		#if not (math.isnan(loss_i) or math.isnan(ci_i) or math.isinf(loss_i) or math.isinf(ci_i)):
		#	loss_sum += loss_i*(len(X)-bs*i)
		#	ci_sum += ci_i*(len(X)-bs*i)	
		#	count += (len(X)-bs*i)


		## print them
		#print('loss={0}'.format(round(loss, 3)))
		#print('ci={0}'.format(round(ci, 3)))
		print(f'loss={loss}')
		print(f'ci={ci}')
		print()
		return ci



	def __vis_layer(self, datasets, dataset_name, flag = 'train'):


		X, time_value, event = zip(*datasets)
		X = np.concatenate(X, axis=0)
		time_value = np.concatenate(time_value, axis=0)
		event = np.concatenate(event, axis=0)
		assert len(X) == len(time_value) == len(event), print ('X, time_value, event len are not equal!!!')

		index = 12
		x_in = X[index]

		vis_out_dir = f'vis_{index}_layer'
		os.makedirs(vis_out_dir, exist_ok=True)

		print (x_in.shape)
		#print (x_in.max(), x_in.min())
		x_in = (x_in-np.min(x_in))/(np.max(x_in)-np.min(x_in))


		io.imsave(os.path.join(vis_out_dir, f'ori.jpg'),  np.squeeze(x_in))

		image_arr=np.reshape(x_in, (-1,160,160,1))
		layer = K.function([self.model.layers[0].input], [self.model.layers[3].output])
		f1 = layer([image_arr])[0]
		re = np.squeeze(np.transpose(f1, (0,3,1,2)))

		print (re.shape)
		for i in range(re.shape[0]):
			temp_im = re[i]
			temp_im = (temp_im-np.min(temp_im))/(np.max(temp_im)-np.min(temp_im))
			io.imsave(os.path.join(vis_out_dir, f'layer2_{i}.jpg'), temp_im)

		return 


	def __vis_layer_cam(self, datasets, dataset_name, flag = 'train'):


		X, time_value, event = zip(*datasets)
		X = np.concatenate(X, axis=0)
		time_value = np.concatenate(time_value, axis=0)
		event = np.concatenate(event, axis=0)
		assert len(X) == len(time_value) == len(event), print ('X, time_value, event len are not equal!!!')

		#index = 1

		for index in range(1148, len(X)):
		
		    x_in = X[index]
		    name = dataset_name[index]
		    #print (name, index)
		    #raise

		    vis_out_dir = f'vis_layer_cam_test'
		    os.makedirs(vis_out_dir, exist_ok=True)

		    print (x_in.shape)
		    print (x_in.max(), x_in.min())
		    #x_in = (x_in-np.min(x_in))/(np.max(x_in)-np.min(x_in))
    
		    io.imsave(os.path.join(vis_out_dir, f'ori_{index}.jpg'),  x_in)


		    image_arr=np.reshape(x_in, (-1,160,160,3))


		    predictions = self.model.predict(image_arr)
		    predicted_class = np.argmax(predictions)
		    print (predictions, predicted_class)

		    #for layer in self.model.layers:
		    #	print (layer.name)
		    #raise
		    cam, heatmap = grad_cam(self.model, image_arr, predicted_class, "conv2d_17")
		    cv2.imwrite(os.path.join(vis_out_dir, f"gradcam_{index}.jpg"), cam)

		    register_gradient()
		    guided_model = modify_backprop(self.model, 'GuidedBackProp')
		    saliency_fn = compile_saliency_function(guided_model)
		    saliency = saliency_fn([image_arr, 0])
		    gradcam = saliency[0] * heatmap[..., np.newaxis]
		    cv2.imwrite(os.path.join(vis_out_dir, f"guided_gradcam_{index}.jpg"), deprocess_image(gradcam))
		    #raise
		return 

