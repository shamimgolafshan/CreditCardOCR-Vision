from imutils import contours
import numpy as np
import argparse
import imutils
import cv2



FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}


def prepare_data(path):
	ref = cv2.imread(path)
	ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
	return cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
	

def find_contours (ref):
	refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	refCnts = imutils.grab_contours(refCnts)
	return contours.sort_contours(refCnts, method="left-to-right")[0]



def reference_contours(refCnts, path_image):
	digits = dict()

	for (i, c) in enumerate(refCnts):
		(x, y, w, h) = cv2.boundingRect(c)
		roi = ref[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		digits[i] = roi

	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

	image = cv2.imread(path_image)
	image = imutils.resize(image, width=300)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

	gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	gradX = gradX.astype("uint8")

	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	thresh = cv2.threshold(gradX, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	return cnts, image, gray, digits

def all_countors(cnts, image, gray, digits):
	locs = list()
	for (i, c) in enumerate(cnts):
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)

		if ar > 2.5 and ar < 4.0:
			if (w > 40 and w < 55) and (h > 10 and h < 20):
				locs.append((x, y, w, h))

	locs = sorted(locs, key=lambda x:x[0])
	output = list()
	# loop over the 4 groupings of 4 digits
	for (i, (gX, gY, gW, gH)) in enumerate(locs):
		# initialize the list of group digits
		groupOutput = []

		group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
		group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		digitCnts = imutils.grab_contours(digitCnts)
		digitCnts = contours.sort_contours(digitCnts,
			method="left-to-right")[0]

		# loop over the digit contours
		for c in digitCnts:
			(x, y, w, h) = cv2.boundingRect(c)
			roi = group[y:y + h, x:x + w]
			roi = cv2.resize(roi, (57, 88))

			scores = []
			for (digit, digitROI) in digits.items():
				result = cv2.matchTemplate(roi, digitROI,
					cv2.TM_CCOEFF)
				(_, score, _, _) = cv2.minMaxLoc(result)
				scores.append(score)

			groupOutput.append(str(np.argmax(scores)))

		cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)

		# update the output digits list
		output.extend(groupOutput)

	return output, image




if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()


	parser.add_argument("-i", "--image_path", required=True,
		help="path to input image")
	parser.add_argument("-r", "--reference_path", required=True,
		help="path to reference OCR-A image")
	args = parser.parse_args()

	
	ref = prepare_data(args.reference_path)
	refCnts	= find_contours (ref)
	cnts, image, gray, digits = reference_contours(refCnts, args.image_path)
	output, image = all_countors(cnts, image, gray, digits)

	cv2.imshow("Image", image)
	cv2.waitKey(0)


	