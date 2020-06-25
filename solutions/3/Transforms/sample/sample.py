from requirements import *
from utils import *
from load_points import *

RX = np.array([[1,0,0],[0,0,-1],[0,1,0]])
RY = np.array([[1,0,0],[0,1,0],[0,0,1]])
RZ = np.array([[0,-1,0],[1,0,0],[0,0,1]])

R = RX.dot(RY)
R = R.dot(RZ)

R = np.array([[0,-1,0],[0,0,-1],[1,0,0]])#,[0,0,0,1]])
T = np.array([[1,0,0,8e-2],[0,1,0,6e-2],[0,0,1,-27e-2]])#,[0,0,0,1]])
K = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],[ 0.000000e+00, 7.215377e+02, 1.728540e+02],[ 0.000000e+00, 0.000000e+00, 1.000000e+00]])
E = R.dot(T)
print(E.shape)
P = K.dot(E)

rgb = cv2.imread(os.path.join('../image.png'))

cmap = plt.cm.get_cmap('hsv', 256)
cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
points = load_velodyne_points('lidar-points.bin')
print(P.shape)
print(points.shape)
arr = []
for i in range(6965):
    arr.append(1)
column_to_be_added = np.array(arr) 
   

def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


# Adding column to numpy array 
result = np.hstack((points, np.atleast_2d(column_to_be_added).T))
print(result.shape)
x = P @ result.transpose()
print(x.shape)

pts_2d = project_to_image(points.transpose(),P)
print(pts_2d.shape)
maxi = -1
for i in range(len(pts_2d.transpose())):
    dist = points[i][0] * points[i][0] + points[i][1]*points[i][1] + points[i][2] * points[i][2]
    maxi = max(dist,maxi)

print("RARA")
for i in range(len(pts_2d.transpose())):
    #print(points[i])
    dist = points[i][0] * points[i][0] + points[i][1]*points[i][1] + points[i][2] * points[i][2]
    color = cmap[int(maxi / dist), :]
    cv2.circle(rgb,(int(pts_2d.transpose()[i][0]),int(pts_2d.transpose()[i][1])),2,tuple(color),-1)
'''
inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]
'''
'''
for i in range(result.shape[1]):
    print(i)
    depth = x[2,i]
    print(depth)
    color = cmap[int(640.0 / depth), :]
    cv2.circle(rgb, (int(np.round(imgfov_pc_pixel[0, i])),int(np.round(imgfov_pc_pixel[1, i]))),2, color=tuple(color), thickness=-1)
'''
'''
points -> (3,n) -> lidar points
result -> (4,n) -> homogenous
P -> (3,4) -> projection matrix
x -> (3,n) -> camera coordinates
'''
plt.imshow(rgb)
plt.savefig('anna.png')
plt.yticks([])
plt.xticks([])
plt.show()
print(rgb.shape)
