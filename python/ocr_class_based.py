from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


class ocrCards:

	def __init__(self,):

		self.path_image = None
		self.FIRST_NUMBER = {
			"3": "American Express",
			"4": "Visa",
			"5": "MasterCard",
			"6": "Discover Card"
		}

	def prepare_data(self, ref_path):
		self.ref = cv2.imread(ref_path)
		self.ref = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
		self.ref = cv2.threshold(self.ref, 10, 255, cv2.THRESH_BINARY_INV)[1]


	def find_contours (self,):
		self.refCnts = cv2.findContours(self.ref.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
		self.refCnts = imutils.grab_contours(self.refCnts)
		self.refCnts = contours.sort_contours(self.refCnts, method="left-to-right")[0]


	def reference_contours(self, path_image):
		self.path_image = path_image
		self.digits = dict()

		for (i, c) in enumerate(self.refCnts):
			(x, y, w, h) = cv2.boundingRect(c)
			roi = self.ref[y:y + h, x:x + w]
			roi = cv2.resize(roi, (57, 88))
			self.digits[i] = roi

		rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
		sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

		self.image = cv2.imread(self.path_image)
		self.image = imutils.resize(self.image, width=300)
		self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

		tophat = cv2.morphologyEx(self.gray, cv2.MORPH_TOPHAT, rectKernel)

		gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
		gradX = np.absolute(gradX)
		(minVal, maxVal) = (np.min(gradX), np.max(gradX))
		gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
		gradX = gradX.astype("uint8")

		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
		thresh = cv2.threshold(gradX, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

		self.cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		self.cnts = imutils.grab_contours(self.cnts)
		return

	def all_countors(self,):
		locs = list()
		for (i, c) in enumerate(self.cnts):
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)

			if ar > 2.5 and ar < 4.0:
				if (w > 30 and w < 55 ) and (h > 10 and h < 20):
					locs.append((x, y, w, h))

		locs = sorted(locs, key=lambda x:x[0])
		output = list()
		# loop over the 4 groupings of 4 self.digits
		for (i, (gX, gY, gW, gH)) in enumerate(locs):
			# initialize the list of group self.digits
			groupOutput = []

			group = self.gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
			group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

			digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
			digitCnts = imutils.grab_contours(digitCnts)
			digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

			# loop over the digit contours
			for c in digitCnts:
				(x, y, w, h) = cv2.boundingRect(c)
				roi = group[y:y + h, x:x + w]
				roi = cv2.resize(roi, (57, 88))

				scores = []
				for (digit, digitROI) in self.digits.items():
					result = cv2.matchTemplate(roi, digitROI,
						cv2.TM_CCOEFF)
					(_, score, _, _) = cv2.minMaxLoc(result)
					scores.append(score)

				groupOutput.append(str(np.argmax(scores)))

			cv2.rectangle(self.image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
			output.extend(groupOutput)

		return output, self.image

	def show_output(self, ref_path, path_image):

		self.prepare_data(ref_path)
		self.find_contours ()
		self.reference_contours(path_image)
		output, image = self.all_countors()
		cv2.imshow("Image", image)
		cv2.waitKey(0)
		


if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--image_path", required=True,
		help="path to input image")
	parser.add_argument("-r", "--reference_path", required=True,
		help="path to reference OCR-A image")
	args = parser.parse_args()
	tmp = ocrCards()
	tmp.show_output(args.reference_path, args.image_path )



