import tensorflow as tf

"""
开mumu模拟器的同时运行这个脚本，可能导致exit code 0xC0000409
应该是GPU被mumu占用了

死磕图片分类器

"""
if __name__ == '__main__':

	# change this as you see fit
	image_path = tf.placeholder(tf.string)
	image_array = tf.image.convert_image_dtype(
			tf.image.decode_png(tf.read_file(image_path), channels=3),
			dtype=tf.uint8)

	label_lines = [line.rstrip() for line
				   in tf.gfile.GFile("D:/MF/tf_files/retrained_labels.txt")]

	with tf.gfile.FastGFile("D:/MF/tf_files/retrained_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.2
	with tf.Session() as sess:
		print("\n----------------------------------------------")
		import time
		start = time.time()
		image = r'test_rose.jpg'
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		image_test = sess.run([image_array], feed_dict={image_path: image})
		predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': image_test[0]})
		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		label = label_lines[top_k[0]]
		print("预测图片为：%s(%s)" % (label, label))
		print('耗时: %.3f' % (time.time() - start))
