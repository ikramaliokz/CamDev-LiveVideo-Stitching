import numpy as np
import cv2 as cv

# Lists to store points
points1 = []
points2 = []

# Flag to toggle between selecting points on frame1 and frame2
toggle_points = True

def click_event(event, x, y, flags, params):
    global points1, points2, toggle_points
    if event == cv.EVENT_LBUTTONDOWN:
        if toggle_points:  # Select point for frame1
            points1.append((x, y))
            print(f"Frame 1: {points1[-1]}")
        else:  # Select point for frame2
            points2.append((x - params[2], y))  # Adjust x for frame2's origin
            print(f"Frame 2: {points2[-1]}")
        cv.circle(params[0], (x, y), 5, (0, 255, 0), -1)
        cv.imshow("Frame Selection", params[0])
        toggle_points = not toggle_points  # Toggle between frame1 and frame2
    elif event == cv.EVENT_RBUTTONDOWN:
        if toggle_points:  # Undo point for frame2
            if points2:
                removed_point = points2.pop()
                print(f"Removed Frame 2: {removed_point}")
        else:  # Undo point for frame1
            if points1:
                removed_point = points1.pop()
                print(f"Removed Frame 1: {removed_point}")
        
        # redraw_points(params[0], params[2])

# def redraw_points(image, frame2_offset):
#     """ Redraws the points on the image after an undo operation. """
#     # cv.destroyAllWindows()
#     image_copy = image.copy()
#     for pt in points1:
#         cv.circle(image_copy, pt, 5, (0, 255, 0), -1)
#     for pt in points2:
#         cv.circle(image_copy, (pt[0] + frame2_offset, pt[1]), 5, (0, 255, 0), -1)
#     cv.imshow("Frame Selection", image_copy)
#     cv.waitKey(0)

def stitch_frames(frame1, frame2, points1, points2):
    src_pts = np.float32(points1)
    dst_pts = np.float32(points2)
    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    h, w = frame1.shape[:2]
    result = cv.warpPerspective(frame1, M, (w * 2, h))
    result[0:frame2.shape[0], 0:frame2.shape[1]] = frame2
    return result

cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(2)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

if not ret1 or not ret2:
    print("Error: Cannot read from camera")
else:
    # Combine both frames into one for display
    combined_frame = np.hstack((frame1, frame2))
    cv.imshow("Frame Selection", combined_frame)
    cv.setMouseCallback("Frame Selection", click_event, [combined_frame, "Frame Selection", frame1.shape[1]])
    cv.waitKey(0)  # Wait for points to be collected

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Error: Cannot read from camera")
            break
        
        if len(points1) >= 4 and len(points2) >= 4:
            stitched_frame = stitch_frames(frame1, frame2, points1, points2)
            cv.imshow("Stitched Video", stitched_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

cap1.release()
cap2.release()
cv.destroyAllWindows()
