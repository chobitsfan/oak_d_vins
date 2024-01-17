#!/usr/bin/env python3

import cv2
import depthai as dai
from collections import deque
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud, ChannelFloat32, Imu
from geometry_msgs.msg import Point32, Vector3
import std_msgs.msg
import time

class FeatureTrackerDrawer:

    lineColor = (200, 0, 200)
    pointColor = (0, 0, 255)
    circleRadius = 2
    maxTrackedFeaturesPathLength = 30
    # for how many frames the feature is tracked
    trackedFeaturesPathLength = 10

    trackedIDs = None
    trackedFeaturesPath = None

    def onTrackBar(self, val):
        FeatureTrackerDrawer.trackedFeaturesPathLength = val
        pass

    def trackFeaturePath(self, features):

        newTrackedIDs = set()
        for currentFeature in features:
            currentID = currentFeature.id
            newTrackedIDs.add(currentID)

            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()

            path = self.trackedFeaturesPath[currentID]

            path.append(currentFeature.position)
            while(len(path) > max(1, FeatureTrackerDrawer.trackedFeaturesPathLength)):
                path.popleft()

            self.trackedFeaturesPath[currentID] = path

        featuresToRemove = set()
        for oldId in self.trackedIDs:
            if oldId not in newTrackedIDs:
                featuresToRemove.add(oldId)

        for id in featuresToRemove:
            self.trackedFeaturesPath.pop(id)

        self.trackedIDs = newTrackedIDs

    def drawFeatures(self, img):

        cv2.setTrackbarPos(self.trackbarName, self.windowName, FeatureTrackerDrawer.trackedFeaturesPathLength)

        for featurePath in self.trackedFeaturesPath.values():
            path = featurePath

            for j in range(len(path) - 1):
                src = (int(path[j].x), int(path[j].y))
                dst = (int(path[j + 1].x), int(path[j + 1].y))
                cv2.line(img, src, dst, self.lineColor, 1, cv2.LINE_AA, 0)
            j = len(path) - 1
            cv2.circle(img, (int(path[j].x), int(path[j].y)), self.circleRadius, self.pointColor, -1, cv2.LINE_AA, 0)

    def __init__(self, trackbarName, windowName):
        self.trackbarName = trackbarName
        self.windowName = windowName
        cv2.namedWindow(windowName)
        cv2.createTrackbar(trackbarName, windowName, FeatureTrackerDrawer.trackedFeaturesPathLength, FeatureTrackerDrawer.maxTrackedFeaturesPathLength, self.onTrackBar)
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()

def getMesh(camSocket, calibData):
    M1 = np.array(calibData.getCameraIntrinsics(camSocket, 1280, 720))
    d1 = np.array(calibData.getDistortionCoefficients(camSocket))
    R1 = np.identity(3)
    new_M1, validPixROI = cv2.getOptimalNewCameraMatrix(M1, d1, (1280,720), 0)
    mapX, mapY = cv2.initUndistortRectifyMap(M1, d1, R1, new_M1, (1280, 720), cv2.CV_32FC1)

    meshCellSize = 16
    mesh0 = []
    # Creates subsampled mesh which will be loaded on to device to undistort the image
    for y in range(mapX.shape[0] + 1): # iterating over height of the image
        if y % meshCellSize == 0:
            rowLeft = []
            for x in range(mapX.shape[1]): # iterating over width of the image
                if x % meshCellSize == 0:
                    if y == mapX.shape[0] and x == mapX.shape[1]:
                        rowLeft.append(mapX[y - 1, x - 1])
                        rowLeft.append(mapY[y - 1, x - 1])
                    elif y == mapX.shape[0]:
                        rowLeft.append(mapX[y - 1, x])
                        rowLeft.append(mapY[y - 1, x])
                    elif x == mapX.shape[1]:
                        rowLeft.append(mapX[y, x - 1])
                        rowLeft.append(mapY[y, x - 1])
                    else:
                        rowLeft.append(mapX[y, x])
                        rowLeft.append(mapY[y, x])
            if (mapX.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)

            mesh0.append(rowLeft)

    mesh0 = np.array(mesh0)
    meshWidth = mesh0.shape[1] // 2
    meshHeight = mesh0.shape[0]
    mesh0.resize(meshWidth * meshHeight, 2)

    mesh = list(map(tuple, mesh0))

    return mesh, meshWidth, meshHeight, new_M1

#def prepare_features_pp(ts, prv_ts, features, prv_features):
#    l_ts = features_left_msg.getTimestamp().total_seconds()
#    dt = l_ts - l_prv_ts
#    tracked_ids = set()
#    if pp_msg is None:
#        pp_msg = PointCloud()
#        pp_msg.header = std_msgs.msg.Header()
#        pp_msg.header.stamp = rospy.Time.from_sec(base_ts + l_ts)
#        pp_msg.header.frame_id = 'map'
#        pp_msg.channels = [ChannelFloat32()]*6
#    for feature in trackedFeaturesLeft:
#        tracked_ids.add(feature.id)
#        x_u = feature.position.x / l_un_fx - l_un_cx / l_un_fx
#        y_u = feature.position.y / l_un_fy - l_un_cy / l_un_fy
#        pp_msg.points.append(Point32(x_u, y_u, 1))
#        pp_msg.channels[0].values.append(feature.id)
#        pp_msg.channels[1].values.append(0)
#        pp_msg.channels[2].values.append(feature.position.x)
#        pp_msg.channels[3].values.append(feature.position.y)
#        vx = 0
#        vy = 0
#        if feature.id in l_tracked_features:
#            vx = (x_u - l_tracked_features[feature.id][0]) / dt
#            vy = (y_u - l_tracked_features[feature.id][1]) / dt
#        pp_msg.channels[4].values.append(vx)
#        pp_msg.channels[5].values.append(vy)
#    pp_pub.publish(pp_msg)
#    l_prv_ts = l_ts
#    no_track_ids = l_tracked_features.keys() - tracked_ids
#    for id in no_track_ids:
#        del l_tracked_features[id]

def create_features_pp(l_prv_features_msg, r_prv_features_msg, l_latest_features_msg, r_latest_features_msg, base_ts, l_un_fx, l_un_fy, l_un_cx, l_un_cy, r_un_fx, r_un_fy, r_un_cx, r_un_cy):
    pp_msg = PointCloud()
    pp_msg.header = std_msgs.msg.Header()
    pp_msg.header.stamp = rospy.Time.from_sec(base_ts + l_latest_features_msg.getTimestamp().total_seconds())
    pp_msg.header.frame_id = 'map'
    pp_msg.channels = [ChannelFloat32()]*6

    prv_features = dict()
    if l_prv_features_msg is not None:
        for feature in l_prv_features_msg.trackedFeatures:
            prv_features[feature.id] = feature.position
        dt = l_latest_features_msg.getTimestamp().total_seconds() - l_prv_features_msg.getTimestamp().total_seconds()
    print("left")
    for feature in l_latest_features_msg.trackedFeatures:
        print(feature.id)
        x_u = feature.position.x / l_un_fx - l_un_cx / l_un_fx
        y_u = feature.position.y / l_un_fy - l_un_cy / l_un_fy
        pp_msg.points.append(Point32(x_u, y_u, 1))
        pp_msg.channels[0].values.append(feature.id)
        pp_msg.channels[1].values.append(0)
        pp_msg.channels[2].values.append(feature.position.x)
        pp_msg.channels[3].values.append(feature.position.y)
        vx = 0
        vy = 0
        if feature.id in prv_features:
            vx = (x_u - prv_features[feature.id].x) / dt
            vy = (y_u - prv_features[feature.id].y) / dt
        pp_msg.channels[4].values.append(vx)
        pp_msg.channels[5].values.append(vy)

    prv_features = dict()
    if r_prv_features_msg is not None:
        for feature in r_prv_features_msg.trackedFeatures:
            prv_features[feature.id] = feature.position
        dt = r_latest_features_msg.getTimestamp().total_seconds() - r_prv_features_msg.getTimestamp().total_seconds()
    print("right")
    for feature in r_latest_features_msg.trackedFeatures:
        print(feature.id)
        x_u = feature.position.x / r_un_fx - r_un_cx / r_un_fx
        y_u = feature.position.y / r_un_fy - r_un_cy / r_un_fy
        pp_msg.points.append(Point32(x_u, y_u, 1))
        pp_msg.channels[0].values.append(feature.id)
        pp_msg.channels[1].values.append(1)
        pp_msg.channels[2].values.append(feature.position.x)
        pp_msg.channels[3].values.append(feature.position.y)
        vx = 0
        vy = 0
        if feature.id in prv_features:
            vx = (x_u - prv_features[feature.id].x) / dt
            vy = (y_u - prv_features[feature.id].y) / dt
        pp_msg.channels[4].values.append(vx)
        pp_msg.channels[5].values.append(vy)

    return pp_msg

def create_pp_msg(ts, cam_id, feature_id, cur_un_pts, x, y, vx, vy):
    pp_msg = PointCloud()
    pp_msg.header = std_msgs.msg.Header()
    pp_msg.header.stamp = rospy.Time.from_sec(ts)
    pp_msg.header.frame_id = 'map'
    pp_msg.channels = [ChannelFloat32()]*6
    pp_msg.points.append(Point32(cur_un_pts[0], cur_un_pts[1], 1))
    pp_msg.channels[0].values.append(feature_id)
    pp_msg.channels[1].values.append(cam_id)
    pp_msg.channels[2].values.append(x)
    pp_msg.channels[3].values.append(y)
    pp_msg.channels[4].values.append(vx)
    pp_msg.channels[5].values.append(vy)
    return pp_msg

base_ts = time.time() - dai.Clock.now().total_seconds()

rospy.init_node('FeatureTracker', anonymous=True, disable_signals=True)
pp_pub = rospy.Publisher("/feature_tracker/feature", PointCloud, queue_size=10)
imu_pub = rospy.Publisher("/camera/imu", Imu, queue_size=50)

l_seq = -1
r_seq = -2
disp_seq = -3
l_prv_features = {}
r_prv_features = {}
prv_features_ts = 0
features_ts = 0

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
featureTrackerLeft = pipeline.create(dai.node.FeatureTracker)
featureTrackerRight = pipeline.create(dai.node.FeatureTracker)
imu = pipeline.create(dai.node.IMU)
depth = pipeline.create(dai.node.StereoDepth)

xoutTrackedFeaturesLeft = pipeline.create(dai.node.XLinkOut)
xoutTrackedFeaturesRight = pipeline.create(dai.node.XLinkOut)
xout_imu = pipeline.create(dai.node.XLinkOut)
xout_disp = pipeline.create(dai.node.XLinkOut)

xoutTrackedFeaturesLeft.setStreamName("trackedFeaturesLeft")
xoutTrackedFeaturesRight.setStreamName("trackedFeaturesRight")
xout_imu.setStreamName("imu")
xout_disp.setStreamName("disparity")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setFps(20)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setFps(20)
monoRight.setCamera("right")

featureTrackerLeft.initialConfig.setNumTargetFeatures(32)
featureTrackerRight.initialConfig.setNumTargetFeatures(32)

# enable ACCELEROMETER_RAW at 500 hz rate
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 250)
# enable GYROSCOPE_RAW at 400 hz rate
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 200)
# it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
# above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
imu.setBatchReportThreshold(1)
# maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
# if lower or equal to batchReportThreshold then the sending is always blocking on device
# useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
imu.setMaxBatchReports(10)

depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(False)
depth.setSubpixel(False)
depth.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)

# Linking
monoLeft.out.link(depth.left)
depth.rectifiedLeft.link(featureTrackerLeft.inputImage)
featureTrackerLeft.outputFeatures.link(xoutTrackedFeaturesLeft.input)

monoRight.out.link(depth.right)
depth.rectifiedRight.link(featureTrackerRight.inputImage)
featureTrackerRight.outputFeatures.link(xoutTrackedFeaturesRight.input)

imu.out.link(xout_imu.input)
depth.disparity.link(xout_disp.input)

# By default the least mount of resources are allocated
# increasing it improves performance
numShaves = 2
numMemorySlices = 2
featureTrackerLeft.setHardwareResources(numShaves, numMemorySlices)
featureTrackerRight.setHardwareResources(numShaves, numMemorySlices)

#featureTrackerConfig = featureTrackerRight.initialConfig.get()
#print("Press 's' to switch between Lucas-Kanade optical flow and hardware accelerated motion estimation!")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    calibData = device.readCalibration()
    intri = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 640, 400)
    fx = intri[0][0]
    cx = intri[0][2]
    fy = intri[1][1]
    cy = intri[1][2]
    l_inv_k11 = 1.0 / fx
    l_inv_k13 = -cx / fx
    l_inv_k22 = 1.0 / fy
    l_inv_k23 = -cy / fy
    intri = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, 640, 400)
    fx = intri[0][0]
    cx = intri[0][2]
    fy = intri[1][1]
    cy = intri[1][2]
    r_inv_k11 = 1.0 / fx
    r_inv_k13 = -cx / fx
    r_inv_k22 = 1.0 / fy
    r_inv_k23 = -cy / fy

    # Output queues used to receive the results
    outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 8, False)
    outputFeaturesRightQueue = device.getOutputQueue("trackedFeaturesRight", 8, False)
    imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    disp_queue = device.getOutputQueue(name="disparity", maxSize=8, blocking=False)

    device.getQueueEvents()

    while True:
        queue_names = device.getQueueEvents(("trackedFeaturesLeft", "trackedFeaturesRight", "imu", "disparity"))
        #print(queue_names)

        for queue_name in queue_names:
            if queue_name == "trackedFeaturesLeft":
                features_data = outputFeaturesLeftQueue.get()
                l_seq = features_data.getSequenceNum()
                l_features = features_data.trackedFeatures
                features_ts = features_data.getTimestamp().total_seconds()
            if queue_name == "trackedFeaturesRight":
                features_data = outputFeaturesRightQueue.get()
                r_seq = features_data.getSequenceNum()
                r_features = features_data.trackedFeatures
            if queue_name == "disparity":
                disp_data = disp_queue.get()
                disp_seq = disp_data.getSequenceNum()
                disp_frame = disp_data.getFrame()
            if queue_name == "imu":
                imuData = imuQueue.get()
                imuPackets = imuData.packets
                for imuPacket in imuPackets:
                    acc = imuPacket.acceleroMeter
                    gyro = imuPacket.gyroscope
                    imu_msg = Imu()
                    imu_msg.header = std_msgs.msg.Header()
                    imu_msg.header.stamp = rospy.Time.from_sec(base_ts + acc.getTimestamp().total_seconds())
                    imu_msg.header.frame_id = 'map'
                    imu_msg.linear_acceleration = Vector3(acc.z, acc.y, -acc.x)
                    imu_msg.angular_velocity = Vector3(gyro.z, gyro.y, -gyro.x)
                    imu_pub.publish(imu_msg)

            if l_seq == r_seq == disp_seq:
                l_seq = -1
                r_seq = -2
                disp_seq = -3
                features = {}
                pp_msg = PointCloud()
                pp_msg.header = std_msgs.msg.Header()
                pp_msg.header.stamp = rospy.Time.from_sec(base_ts + features_ts)
                pp_msg.header.frame_id = 'map'
                pp_msg.channels = [ChannelFloat32(), ChannelFloat32(), ChannelFloat32(), ChannelFloat32(), ChannelFloat32(), ChannelFloat32()]
                for feature in l_features:
                    x = feature.position.x
                    y = feature.position.y
                    cur_un_pts = (l_inv_k11 * x + l_inv_k13, l_inv_k22 * y + l_inv_k23)
                    features[feature.id] = cur_un_pts
                    row = round(y)
                    col = round(x)
                    if col > 639:
                        col = 639
                    if row > 399:
                        row = 399
                    disp = disp_frame[row, col]
                    if disp > 0:
                        for r_feature in r_features:
                            dy = y - r_feature.position.y
                            dx = x - disp - r_feature.position.x
                            if dy * dy + dx * dx <= 36: #pair found
                                dt = features_ts - prv_features_ts
                                vx = 0
                                vy = 0
                                id = feature.id
                                if id in l_prv_features:
                                    vx = (cur_un_pts[0] - l_prv_features[id][0]) / dt
                                    vy = (cur_un_pts[1] - l_prv_features[id][1]) / dt
                                pp_msg.points.append(Point32(cur_un_pts[0], cur_un_pts[1], 1))
                                pp_msg.channels[0].values.append(id)
                                pp_msg.channels[1].values.append(0)
                                pp_msg.channels[2].values.append(x)
                                pp_msg.channels[3].values.append(y)
                                pp_msg.channels[4].values.append(vx)
                                pp_msg.channels[5].values.append(vy)

                                x = r_feature.position.x
                                y = r_feature.position.y
                                cur_un_pts = (r_inv_k11 * x + r_inv_k13, r_inv_k22 * y + r_inv_k23)
                                vx = 0
                                vy = 0
                                id = r_feature.id
                                if id in r_prv_features:
                                    vx = (cur_un_pts[0] - r_prv_features[id][0]) / dt
                                    vy = (cur_un_pts[1] - r_prv_features[id][1]) / dt
                                pp_msg.points.append(Point32(cur_un_pts[0], cur_un_pts[1], 1))
                                pp_msg.channels[0].values.append(feature.id)
                                pp_msg.channels[1].values.append(1)
                                pp_msg.channels[2].values.append(x)
                                pp_msg.channels[3].values.append(y)
                                pp_msg.channels[4].values.append(vx)
                                pp_msg.channels[5].values.append(vy)
                                break
                pp_pub.publish(pp_msg)
                l_prv_features = features
                prv_features_ts = features_ts
                r_prv_features = {}
                for feature in r_features:
                    r_prv_features[feature.id] = (r_inv_k11 * feature.position.x + r_inv_k13, r_inv_k22 * feature.position.y + r_inv_k23)
#                seq = features_msg.getSequenceNum()
#                features = features_msg.trackedFeatures
#                ts = features_msg.getTimestamp().total_seconds()
#                dt = ts - l_prv_ts
#                tracked_ids = set()
#                if pp_msg is None:
#                    pp_msg = PointCloud()
#                    pp_msg.header = std_msgs.msg.Header()
#                    pp_msg.header.stamp = rospy.Time.from_sec(base_ts + l_ts)
#                    pp_msg.header.frame_id = 'map'
#                    pp_msg.channels = [ChannelFloat32()]*6
#                for feature in trackedFeaturesLeft:
#                    tracked_ids.add(feature.id)
#                    x_u = feature.position.x / l_un_fx - l_un_cx / l_un_fx
#                    y_u = feature.position.y / l_un_fy - l_un_cy / l_un_fy
#                    pp_msg.points.append(Point32(x_u, y_u, 1))
#                    pp_msg.channels[0].values.append(feature.id)
#                    pp_msg.channels[1].values.append(0)
#                    pp_msg.channels[2].values.append(feature.position.x)
#                    pp_msg.channels[3].values.append(feature.position.y)
#                    vx = 0
#                    vy = 0
#                    if feature.id in l_tracked_features:
#                        vx = (x_u - l_tracked_features[feature.id][0]) / dt
#                        vy = (y_u - l_tracked_features[feature.id][1]) / dt
#                    pp_msg.channels[4].values.append(vx)
#                    pp_msg.channels[5].values.append(vy)
#                pp_pub.publish(pp_msg)
#                l_prv_ts = l_ts
#                no_track_ids = l_tracked_features.keys() - tracked_ids
#                for id in no_track_ids:
#                    del l_tracked_features[id]

            #print("seq", features_left_msg.getSequenceNum(), features_right_msg.getSequenceNum())
        #rightFeatureDrawer.trackFeaturePath(trackedFeaturesRight)
        #rightFeatureDrawer.drawFeatures(rightFrame)

        # Show the frame
        #cv2.imshow(leftWindowName, leftFrame)
        #cv2.imshow(rightWindowName, rightFrame)

        #key = cv2.waitKey(1)
        #if key == ord('q'):
        #    break
        #elif key == ord('s'):
        #    if featureTrackerConfig.motionEstimator.type == dai.FeatureTrackerConfig.MotionEstimator.Type.LUCAS_KANADE_OPTICAL_FLOW:
        #        featureTrackerConfig.motionEstimator.type = dai.FeatureTrackerConfig.MotionEstimator.Type.HW_MOTION_ESTIMATION
        #        print("Switching to hardware accelerated motion estimation")
        #    else:
        #        featureTrackerConfig.motionEstimator.type = dai.FeatureTrackerConfig.MotionEstimator.Type.LUCAS_KANADE_OPTICAL_FLOW
        #        print("Switching to Lucas-Kanade optical flow")

        #    cfg = dai.FeatureTrackerConfig()
        #    cfg.set(featureTrackerConfig)
        #    inputFeatureTrackerConfigQueue.send(cfg)
