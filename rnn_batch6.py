import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesData():

	def __init__ (self,num_points,xmin,xmax):
		self.xmin = xmin
		self.xmax = xmax
		self.num_points = num_points
		self.resolution = (xmax-xmin)/num_points
		self.x_data = np.linspace(xmin,xmax,num_points)
		self.y_true = np.sin(self.x_data)

	def next_batch(self,batch_size,steps,return_batch_ts=False):
		#Grab random starting point
		random_start = np.random.rand(batch_size,1)
       
		#convert to be on time series
		ts_start = random_start * (self.xmax - self.xmin -(steps*self.resolution))

		#Create batch time series
		batch_ts = ts_start + np.arange(0.0,steps+1)*self.resolution

		#Create Y values
		y_batch = np.sin(batch_ts)

		if return_batch_ts:
			return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1) , batch_ts
		else:
			return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1)

ts_data = TimeSeriesData(250,0,10)
plt.plot(ts_data.x_data , ts_data.y_true)
plt.show()


num_time_steps = 30
y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)
#print(ts.flatten().shape)
plt.plot(ts.flatten()[1:],y2.flatten(),'*')
plt.show()



plt.plot(ts_data.x_data,ts_data.y_true,label='Sin(t)')
plt.plot(ts.flatten()[1:],y2.flatten(),'*',label="A Single Training Instance")
plt.show()

#Training data
train_instance = np.linspace(5,5+ts_data.resolution*(num_time_steps+1),num_time_steps+1)
print(train_instance)

plt.title('A Training Instance')

plt.plot(train_instance[:-1],np.sin(train_instance[:-1]),'bo',markersize = 15, alpha = 0.5, label="Instance")
plt.plot(train_instance[1:],np.sin(train_instance[1:]),'ko',markersize = 12, alpha = 0.5, label="Target")
plt.legend()
plt.show()






#Creating the model
tf.reset_default_graph()

num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.001
epochs = 2000
batch_size = 1


X = tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])


#RNN CELL LAYER
cell = tf.contrib.rnn.BasicRNNCell(num_units = num_neurons, activation = tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_size=num_outputs)
outputs , states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)








loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#Session
with tf.Session() as sess:
	sess.run(init)
	for i in range(epochs):
		x_batch,y_batch = ts_data.next_batch(batch_size,num_time_steps)
		sess.run(train,{X:x_batch , y:y_batch})
		if i % 100 == 0:
			mse = loss.eval({X:x_batch , y:y_batch})
			print(mse)
	saver.save(sess,'./rnn_timeseries')







#Prediction
with tf.Session() as sess:
	saver.restore(sess,'./rnn_timeseries')
	x_new = np.sin(np.array(train_instance[:-1].reshape(-1,num_time_steps,num_inputs)))
	y_pred = sess.run(outputs,feed_dict={X:x_new})

plt.title('Testing the Model')
#Training Instance
plt.plot(train_instance[:-1],np.sin(train_instance[:-1]),'bo',markersize=15,alpha=0.5,label='Training Instance')
#target to predict (correct values for np.sin)
plt.plot(train_instance[1:],np.sin(train_instance[1:]),'ko',markersize=10,label='Target')
#Model Predictions
plt.plot(train_instance[1:],y_pred[0,:,0],'r.',markersize=10,label='Predictions')

plt.legend()
plt.tight_layout()
plt.show()