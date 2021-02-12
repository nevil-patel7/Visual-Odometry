import cv2
import numpy as np
import random
from UndistortImage import UndistortImage
from ReadCameraModel import ReadCameraModel
import glob


class Process:
    def __init__(self):
        dataset = glob.glob("Oxford_dataset/stereo/centre/*.png")
        dataset.sort()
        self.dataset_images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH) for img in dataset]
        self.fx, self.fy, self.cx, self.cy, self.G_camera_image1, self.LUT1 = ReadCameraModel('Oxford_dataset/model')
        self.fx2, self.fy2, self.cx2, self.cy2, self.G_camera_image2, self.LUT2 = ReadCameraModel('Oxford_dataset/model')
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def return_dataset(self):
        return self.dataset_images


    def preprocess(self,image,nextimage):
        rgb1 = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR)
        rgb2 = cv2.cvtColor(nextimage, cv2.COLOR_BAYER_GR2BGR)
        undimg_1 = UndistortImage(rgb1,self.LUT1)
        undimg_1 = cv2.cvtColor(undimg_1, cv2.COLOR_BGR2GRAY)
        undimg_2 = UndistortImage(rgb2, self.LUT2)
        undimg_2 = cv2.cvtColor(undimg_2, cv2.COLOR_BGR2GRAY)
        imgeqi_1 = cv2.equalizeHist(undimg_1)
        # imgeqi_1 = (imgeqi_1[0:800, 0:1280].copy())
        imgeqi_2 = cv2.equalizeHist(undimg_2)
        # imgeqi_2 = (imgeqi_2[0:800, 0:1280].copy())


        return imgeqi_1,imgeqi_2


    def Fundamental_matrix(self, x1, x2):
        n = x1.shape[1]
        A = np.zeros((n, 9))
        for i in range(n):
            A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i],
                    x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i],
                    x2[0, i], x2[1, i], 1]

        U, S, V = np.linalg.svd(A)
        F = V[-1].reshape(3, 3)
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = np.dot(U, np.dot(np.diag(S), V))
        return F

    def RT_matrix(self, F,)\
            :
        E = np.dot(self.K.T, np.dot(F, self.K))
        U, S, V = np.linalg.svd(E)
        E = np.dot(U, np.dot(np.diag([1, 1, 0]), V))
        U, S, V = np.linalg.svd(E)
        W = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
        t1 = U[:, 2]
        t2 = -U[:, 2]
        t3 = U[:, 2]
        t4 = -U[:, 2]
        R1 = np.dot(U, np.dot(W, V))
        R2 = np.dot(U, np.dot(W, V))
        R3 = np.dot(U, np.dot(W.T, V))
        R4 = np.dot(U, np.dot(W.T, V))
        if np.linalg.det(R1) < 0:
            R1 = -R1
            t1 = -t1
        elif np.linalg.det(R2) < 0:
            R2 = -R2
            t2 = -t2
        elif np.linalg.det(R3) < 0:
            R3 = -R3
            t3 = -t3
        elif np.linalg.det(R4) < 0:
            R4 = -R4
            t4 = -t4
        P = [np.vstack((R1, t1)).T,
             np.vstack((R2, t2)).T,
             np.vstack((R3, t3)).T,
             np.vstack((R4, t4)).T]
        return P

    def R_PTS_EIGHT(self, k_pts_m):
        mat_1 = [[[], [], [], [], [], [], [], []],
                 [[], [], [], [], [], [], [], []],
                 [[], [], [], [], [], [], [], []],
                 [[], [], [], [], [], [], [], []],
                 [[], [], [], [], [], [], [], []],
                 [[], [], [], [], [], [], [], []],
                 [[], [], [], [], [], [], [], []],
                 [[], [], [], [], [], [], [], []]]

        resolution_X = 160
        resolution_Y = 120
        for index, kpts1, kpts2 in k_pts_m:
            x_cell = int(kpts1[0] / resolution_X)
            y_cell = int(kpts1[1] / resolution_Y)
            mat_1[x_cell][y_cell].append((kpts1[0], kpts1[1], kpts2[0], kpts1[1]))
        best_inliers = np.array([])
        cell = []
        count = 0
        best_F = np.array([])

        for i in range(8):
            for j in range(8):
                if len(mat_1[i][j]) != 0:
                    cell.append(mat_1[i][j])

        while (count <= 100):

            eight_points = []

            eight_cells = random.sample(cell, k=8)

            # obtaining eight points list1
            for i in eight_cells:
                a = random.choice(i)
                eight_points.append(a)

            img_pts_1 = []
            img_pts_2 = []
            for x1, y1, x2, y2 in eight_points:
                img_pts_1.append((x1, y1))
                img_pts_2.append((x2, y2))

            img_pts_1 = np.array(img_pts_1)
            img_pts_2 = np.array(img_pts_2)

            F = self.Fundamental_matrix(img_pts_1.T, img_pts_2.T)

            inliers = []

            for index, kpts1, kpts2 in k_pts_m:

                x_1 = kpts1[0]
                y_1 = kpts1[1]
                x_2 = kpts2[0]
                y_2 = kpts2[1]

                d1 = np.dot(F, np.array([x_1, y_1, 1]))
                d2 = np.dot(F.T, np.array([x_1, y_1, 1]))
                error = np.linalg.norm(np.abs(np.dot(np.array([x_2, y_2, 1]).T, d1))) / np.sqrt(
                    np.dot(d1.T, d1) + np.dot(d2.T, d2))

                if error <= 0.5:
                    inliers.append((x_1, y_1, x_2, y_2))

            if len(inliers) > len(best_inliers):
                best_inliers = np.array([])
                best_F = np.array([])
                best_inliers = np.array(inliers)
                best_F = F

            count += 1

        return best_F, best_inliers

    def TR_pts(self, x1, x2, P1, P2):
        M = np.zeros((6, 6))
        M[:3, :4] = P1
        M[:2, 4] = -x1
        M[2, 4] = 1
        M[3:, :4] = P2
        M[3:5, 5] = -x2
        M[5, 5] = 1
        U, S, V = np.linalg.svd(M)
        X = V[-1, :4]
        return X / X[3]

    def pts_three(self, x1, x2, P1, P2):
        n = x1.shape[1]
        X = [self.TR_pts(x1[:, i], x2[:, i], P1, P2) for i in range(n)]
        return np.array(X).T
