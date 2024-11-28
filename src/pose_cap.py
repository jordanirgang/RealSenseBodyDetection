from python_path import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
image_width = 640
image_height = 480
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def draw_point(img,pose):
    try:
        max_depth = 10000;
        x_img = int(pose.x*image_width)
        y_img = int(pose.y*image_height)

        z_img = (1/(aligned_depth_frame.get_distance(x_img,y_img)))*100 # aligned_depth_frame is a 640x480 depth image


        cv2.circle(img,(x_img,y_img), int(z_img), (0,0,255), -1)
    finally:
        return img


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
frame_count = 0
# Streaming loop
while True:
    frame_count +=1
    print(("frame_count",frame_count))
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
     #frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Render images:
    #   depth align to color on left
    #   depth on right
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


    print("starting annotate")


    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:


        #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
        results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        mp_drawing.draw_landmarks(
        color_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #skip display if no poses
    if not results.pose_landmarks:
      continue
    print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
    depth_colormap = draw_point(depth_colormap,results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]);
    depth_colormap = draw_point(depth_colormap,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]);
    depth_colormap = draw_point(depth_colormap,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]);


    
    images = np.hstack((color_image, depth_colormap))

    #images = np.hstack((bg_removed, depth_colormap))
    cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
    cv2.imshow('Align Example', images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        pipeline.stop()
        break
