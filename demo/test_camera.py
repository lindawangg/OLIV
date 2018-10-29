import numpy as np 
import cv2

cap = cv2.VideoCapture(-1)

while (True):
	ret, frame = cap.read()
	#output_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	if ret:
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()