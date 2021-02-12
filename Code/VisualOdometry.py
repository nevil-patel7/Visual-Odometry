
from functions import *

x_plot = []
z_plot = []

newProcess = Process()
dataset_images=newProcess.return_dataset()
jointlist = []
a = 0
N = np.zeros((4, 4))
h = np.eye(4)
c_position = np.zeros((3, 1))
c_rotation = np.eye(3)
while a < (len(dataset_images) - 1):
    preprocessed_image1,preprocessed_image2=newProcess.preprocess(dataset_images[a],dataset_images[a+1])

    # Initiate STAR detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(preprocessed_image1, None)
    kp2, des2 = orb.detectAndCompute(preprocessed_image2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    good_points = []
    for match in matches:
        good_points.append(match)
    # print(good_points)

    pts1 = np.float32([kp1[match.queryIdx].pt for match in good_points])
    pts2 = np.float32([kp2[match.trainIdx].pt for match in good_points])

    # to overlay good features
    # for i in range(0, len(pts1)):
    #     pos = pts1[i]
    #     b = cv2.circle(dataset_images[a], tuple(pos), 3, color=(0, 0, 255))
    #     cv2.imwrite('points.png', b)

    for i in range(len(pts1)):
        jointlist.append((i, pts2[i], pts1[i]))  # CHANGE

    BF, inliers = newProcess.R_PTS_EIGHT(jointlist)
    in1 = []
    in2 = []
    for q1, w1, q2, w2 in inliers:
        in1.append((q1, w1))
        in2.append((q2, w2))
    in1 = np.array(in1)
    # print(in1.shape)
    in2 = np.array(in2)
    BF = newProcess.Fundamental_matrix(in1.T, in2.T)
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = newProcess.RT_matrix(BF)
    indices = 0
    max_res = 0
    for i in range(4):
        X = newProcess.pts_three(in1.T, in2.T, P1, P2[i])
        d1 = np.dot(P1, X)[2]
        d2 = np.dot(P2[i], X)[2]
        if np.sum(d1 > 0) + np.sum(d2 > 0) > max_res:
            max_res = np.sum(d1 > 0) + np.sum(d2 > 0)
            indices = i
            infront = (d1 > 0) & (d2 > 0)

    P = P2[indices]
    Rot = P[:3, :3]
    Trans = P[:3, 3]

    c_position = c_position + c_rotation.dot(Trans)
    c_rotation = Rot.dot(c_rotation)

    x_plot.append(-c_position[0, 0])
    z_plot.append(-c_position[2, 0])
    print(a)
    a += 1
    jointlist = []
final = []
for i in range(len(x_plot)):
    final.append((x_plot[i], z_plot[i]))
# print(final)
plt.scatter(x_plot, z_plot)
plt.show()
