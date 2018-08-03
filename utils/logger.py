# coding=utf-8
import os
LOG_FILE = os.path.dirname(__file__) + r'/../log/log_%s.txt'


class Logger(object):

	_instance = None

	@staticmethod
	def instance():
		if not Logger._instance:
			Logger._instance = Logger()
		return Logger._instance
		pass

	def __init__(self):
		dirname = os.path.dirname(LOG_FILE)
		if not os.path.exists(dirname):
			os.mkdir(dirname)
		import time
		self.log = open(LOG_FILE % time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "a", 512)
		pass

	def info(self, *args):
		msg = ""
		if args:
			for arg in args:
				msg += " " + str(arg)
		self.log.write(msg + '\r\n')
		self.log.flush()
		pass
	pass



if __name__ == '__main__':
	log = Logger.instance()
	log.info("test log")
	pass

