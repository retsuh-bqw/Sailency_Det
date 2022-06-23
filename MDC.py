import math
import itertools
import cv2 as cv
import numpy as np
import sys
import os
import matplotlib.pyplot as plt


class MDC():
	def __init__(self):

		# self.saillency_map = self.get_sailency_map(cv.imread(src_img))
		pass


	def normalize_range(self, src ,begin=0 , end=255):
		# 规范化数据范围大小
		dst = np.zeros((len(src), len(src[0])))
		min_, max_ = np.amin(src), np.amax(src)
		for y, x in itertools.product(range(len(src)), range(len(src[0]))):
			if min_ != max_:
				dst[y][x] = (src[y][x] - min_) * (end - begin) / (max_ - min_) + begin
			else:
				dst[y][x] = (end + begin) / 2
		return dst

	def normalize(self, src):
		# 根据数据情况进行拉伸或者平滑
		src = self.normalize_range(src, 0., 1.)  # 将像素值归一化到0-1之间
		max_array = np.amax(src)  # amax：每一个元素中的最大值找到，然后把这些最大值找出来组成一个数组
		maxs = []
		for y in range(1, len(src) - 1):
			for x in range(1, len(src[0]) - 1):
				val = src[y][x]
				if val == max_array:
					continue
				if val > src[y - 1][x] and val > src[y + 1][x] and val > src[y][x - 1] and val > src[y][x + 1]:
					# 当前点亮度大于上下左右
					maxs.append(val)
		if len(maxs) != 0:
			src *= math.pow(max_array - (np.sum(maxs) / np.float64(len(maxs))), 2.)
		return src

	#计算原始显著性图
	def saliencyraw(self, src):
		W=src.shape[0]
		H=src.shape[1]
		W1=W+1
		H1=H+1

		MAP_SQR = 0
		MAP_CH0 = 1
		MAP_CH1 = 2
		MAP_CH2 = 3
		MAP_GROUP = 4

		Intmap = np.zeros((W1,H1,MAP_GROUP),int)#保存积分图像
		for y in range (1,H1):
			x=0
			py=y-1   #当前像素在第几列
			px=x    #当前像素是第几行的像素
			sumc=0
			sumd0=0
			sumd1=0
			sumd2=0

			qy=y  #当前扫描像素的下一列
			qx=0  #是第几行
			Intmap[qx][qy][MAP_SQR] = 0
			Intmap[qx][qy][MAP_CH0] = 0
			Intmap[qx][qy][MAP_CH1] = 0
			Intmap[qx][qy][MAP_CH2] = 0
			qx = qx + 1
			#求积分图像 行 * 列 * 4 保存r ^ 2 + g ^ 2 + b ^ 2以及r和、g和、b和
			for x in range (1,W1):
				#计算输入图像中的此列第一行到当前像素所有像素的r、g、b和信息
				sumc = sumc + src[px][py][0]**2+src[px][py][1]**2+src[px][py][2]**2
				sumd0 = sumd0 + src[px][py][0]
				sumd1 = sumd1 + src[px][py][1]
				sumd2 = sumd2 + src[px][py][2]

				#新值等于同行相邻像素的和加上同列之前的和
				#用Intmap的除去最后一行、第一列保存积分图
				Intmap[qx][qy][MAP_SQR] = Intmap[qx][qy - 1][MAP_SQR] + sumc
				Intmap[qx][qy][MAP_CH0] = Intmap[qx][qy - 1][MAP_CH0] + sumd0
				Intmap[qx][qy][MAP_CH1] = Intmap[qx][qy - 1][MAP_CH1] + sumd1
				Intmap[qx][qy][MAP_CH2] = Intmap[qx][qy - 1][MAP_CH2] + sumd2

				px = px + 1 #计算此列下一个元素
				qx = qx + 1 #填充下一行的下一个元素

		salimg = np.zeros((W, H),float)
		ctbl0 = 0
		ctbl1 = 0
		ctbl2 = 0
		ctbl3 = 0
		for y in range (1,H1):
			x = 0
			px = x
			py = y - 1
			for x in range (1,W1):
				C = src[px][py][0]**2 + src[px][py][1]**2 + src[px][py][2]**2
				d0 = 2*src[px][py][0]
				d1 = 2*src[px][py][1]
				d2 = 2*src[px][py][2]

				X_Y = [(x,y), (W1-1,y), (x,H1-1), (W1-1,H1-1)]


				ctbl0, ctbl1, ctbl2, ctbl3 = [Intmap[qx][qy][MAP_SQR] - Intmap[qx][qy][MAP_CH0] * d0 \
											- Intmap[qx][qy][MAP_CH1] * d1 - Intmap[qx][qy][MAP_CH2] \
											* d2 + qx * qy * C for qx, qy in X_Y]


				ctbl1 = ctbl1 - ctbl0 
				ctbl2 = ctbl2 - ctbl0
				ctbl3 = ctbl3 - ctbl1 - ctbl2 - ctbl0

				min1 = min(ctbl0,ctbl1)
				min2 = min(ctbl2,ctbl3)
				min_sal = min(min1,min2) 

				#形成方向对比度图
				if min_sal<0:
					min_sal = 0

				salimg[x-1][y-1]= math.sqrt(min_sal)
				#更改当前像素位置
				px = px+1

		return self.normalize(salimg)


	def saliency_smooth(self, src , rawsal):
		N = 24
		borderCntTbl = np.zeros([N**3],int)
		colorCntTbl = np.zeros([N**3],int)
		salSmoothTbl = np.zeros([N**3],float)
		r = 20
		W = src.shape[0]
		H = src.shape[1]
		pidxmap = np.zeros((W,H),int)

		for y in range (0,H):

			px = 0
			py = y
			psal = rawsal

			qx = 0
			qy = y
			for x in range (0,W):
				a = int (src[px][py][0] * N / 256)
				b = int (src[px][py][1] * N / 256)
				c = int (src[px][py][2] * N / 256)
				idx = int(a * N * N + b *N +c)
				pidxmap[x][y] = idx
				colorCntTbl[idx] = colorCntTbl[idx] + 1
				salSmoothTbl[idx] = salSmoothTbl[idx] + psal[qx][qy]

				if (y < r or x < r or x > W - r):
					borderCntTbl[idx] = borderCntTbl[idx]+1
				px = px + 1
				qx = qx + 1

		for i in range (0,N**3):
			if (colorCntTbl[i] > 0):
				salSmoothTbl[i] = salSmoothTbl[i]/colorCntTbl[i]
				ratiobd = borderCntTbl[i] / r / math.sqrt(colorCntTbl[i])
				if (ratiobd > 0.01 ):
					salSmoothTbl[i] = salSmoothTbl[i]*math.exp(-3.0 * ratiobd)

		for y in range (0,H):
			qx = 0
			qy = y
			for x in range (0,W):
				idx = pidxmap[x][y]
				psal[qx][qy] = (psal[qx][qy] + salSmoothTbl[idx])/2
				qx = qx + 1

		return psal

	# 大津法
	def threshold_calc(self, smoothsal):
		W = smoothsal.shape[0]
		H = smoothsal.shape[1]
		threshold = 0

		pixelcount = np.zeros(256)
		pixelpro = np.zeros(256)

		r = 0.03
		totalcount = 0
		for i in range(math.ceil(W*r) , int(W*(1-r))):
			for j in range (math.ceil(H*r),int(H*(1-r))):

				pixelcount[int(smoothsal[i][j]*255)] += 1
				totalcount +=1

		utmp=0
		usum=0
		for i in range (0,256):
			pixelpro[i]=float(pixelcount[i]/totalcount)
			utmp += i*pixelpro[i]
			usum += i*i*pixelpro[i]

		w0 = 0
		u0tmp = 0
		u0sum = 0
		deltamax = 0

		for i in range (0,256):
			w0 += pixelpro[i]
			u0tmp += i*pixelpro[i]
			u0sum += i*i*pixelpro[i]

			w1 = 1-w0
			u1tmp = utmp - u0tmp
			u1sum = usum - u0sum

			u0 = u0tmp / w0
			u1 = u1tmp / (w1+1e-6)
			u = u0tmp + u1tmp

			deltatmp = w0 *(u0-u)*(u0-u)+w1*(u1-u)*(u1-u)
			if (deltatmp>deltamax):
				deltamax = deltatmp
				threshold = i

		return float(threshold / 255)


	def saliency_enhance(self, src , smoothsal):
		sal = smoothsal
		W = src.shape[0]
		H = src.shape[1]
		theta = 0.5
		thres_val = self.threshold_calc(sal)

		markers = np.zeros((W,H),int)
		markers = markers.astype('uint8')
		thtop = max(min(0.9,thres_val*(1+theta)),0.3)
		thbottom = min(max(0.1,thres_val*(1-theta)),0.3)
		m_fg = 0x40
		m_bg = 0x20

		cntfg = 0
		for i in range (0,W):
			#markers矩阵的指针
			qx = i
			qy = 0
			#smoothsal的指针
			px = i
			py = 0
			for j in range(0,H):
				if(sal[px][py] > thtop):
					cntfg +=1
					markers[qx][qy] = m_fg
				elif (sal[px][py]<thbottom):
					markers[qx][qy] = m_bg
				else:
					markers[qx][qy] = 0
				qy += 1
				py += 1
		bgerode = min(int(math.sqrt(cntfg)*0.05),3)

		if bgerode > 0:
			element = cv.getStructuringElement(cv.MORPH_RECT, (1 + bgerode * 2, 1 + bgerode * 2))
			markers = markers.astype('uint8')
			markers = cv.erode(markers , element)
			markers = cv.dilate(markers , element)

		img = src
		alpha = 0.05
		beta = 0.2
		for i in range (0,W):

			qx = i
			qy = 0

			px = i
			py = 0
			for j in range (0,H):
				index = markers[qx][qy]
				if (index == m_bg):
					sal[px][py] = sal[px][py] * beta
				elif (index == m_fg):
					sal[px][py] = (1 - alpha) + sal[px][py] * alpha
				qy += 1
				py += 1
		return sal


	def get_sailency_map(self, src_img):
		src = cv.medianBlur(src_img,3)
		src = cv.cvtColor(src, cv.COLOR_BGR2Lab)
		src = np.array(src)
		rawsal = self.saliencyraw(src) 
		smoothsal = self.saliency_smooth(src , rawsal)
		enhancesal = self.saliency_enhance(src , smoothsal)

		return smoothsal * 255


def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index


if __name__ == '__main__':

    from utils.plot import demo_plot
    demo_plot('./data/images/test', 'MDC_4.png', saillency_method = MDC())


