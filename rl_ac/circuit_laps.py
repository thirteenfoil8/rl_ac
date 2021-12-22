import time

import numpy as np
import os
#from os import listdir
#from os.path import isfile, join, splitext
import glob
from csv import reader
import math
from scipy.interpolate import interp1d
# IMPORTANT requires matplotlib 3.3.3 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.transforms as mtransforms
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.widgets import AxesWidget, CheckButtons, RadioButtons, Slider

global laplines, checkbtn, plotter, LINE_WIDTH, LINE_ALPHA, LINE_COLOR

TRACK = "barcelona"
#TRACK = "zandvoort"

FOLDER = "Kevin_vs_Nicolas_2"
#FOLDER = "Alain"
#FOLDER = "Kevin"
#FOLDER = "Max"

FOLDER = TRACK + "/" + FOLDER


#FILE_FILTER = ["20201129", "20201128", "20201127"]         # filter with pattern in the filename
#FILE_FILTER = ["20201201"]
FILE_FILTER = ["20201214"]

DISPLAY_ONLY_FULL_LAPS = True       # display only full laps, from checker flag line to checker flag line (can be a valid or not lap)
DISPLAY_LAP_TIME = False             # display the lap time in the command line interface
HIDE_ALL_NOT_BEST_OR_WORST = True   # plot only the best and worst lap, other laps can be displayed with the checkbuttons
DISPLAY_ONLY_LAP_DELTA_MAX = 0    # filter out laps with a delta time over the defined value in seconds, set to 0 for no filtering

# Display Modes
DISPLAY_MODE_LAPTIME = 0
DISPLAY_MODE_PEDALS = 1
DISPLAY_MODE_VELOCITY = 2
DISPLAY_MODE_CURVATURE = 3
DISPLAY_MODE_VELOCITY_PLOT = 4
DISPLAY_MODE_G_PLOT = 5
DISPLAY_MODE_TRAJ_DIST = 6
DISPLAY_MODE_VELOC_ANGLE = 7
DISPLAY_MODE_TIRE_SLIP_ANGLE = 8
DISPLAY_MODE_TIRE_SLIP_RATIO = 9
DISPLAY_MODE_LIDAR = 10


DISPLAY_MODE = DISPLAY_MODE_PEDALS



CIRCUIT_LIMIT_X = 800       # in meters     800 for Bacerlone
CIRCUIT_LIMIT_Y = 300       # in meters     300 for Bacerlone
CIRCUIT_ROTATION = -57      # in degrees   -57° for Bacerlone
CIRCUIT_TRANSLATE_X = 0     # in meters       0 for Bacerlone
CIRCUIT_TRANSLATE_Y = 0     # in meters       0 for Bacerlone

LINE_WIDTH = 2.0              # default plot line width
LINE_ALPHA = 0.3              # default plot line transparency (alpha)
LINE_COLOR = (0.3, 0.3, 1.0)  # default plot line color in RGB


dir_path = os.path.dirname(os.path.realpath(__file__))
input_path = dir_path + "/" + FOLDER + "/"
track_path = dir_path + "/" + TRACK + "/"

trajectories = []
valid_trajectories = []
valid_tp_trajectories = []
filename_array = []
laplines = []

track_data = []
track_length = 0
track_filename = TRACK + ".csv"

# arrays of 3D world coordinate frame vertors
centerline_xyz = []
sideleft_xyz = []
sideright_xyz = []
direction_xyz = []

# arrays of 2D projected world coordinate frame vertors   (x2d = x3d   y2d = - z3d)
centerline_xy = []
sideleft_xy = []
sideright_xy = []
direction_xy = []



# Load track data 
# 20
#['Position_X', 'Position_Y', 'Position_Z', 'Length', 'SideLeft', 'SideRight', 'Radius', 'Speed', 'Brake', 'Camber', 'Direction', 'Grade', 
# 'Normal_X', 'Normal_Y', 'Normal_Z', 'Forward_X', 'Forward_Y', 'Forward_Z', 'Tag', 'LagG_Obsolete']
if os.path.exists(track_path + track_filename):
    first_line = True
    array_size = 0
    
    with open(track_path + track_filename, newline='') as csvfile:
        csvreader = reader(csvfile, delimiter=',', quotechar='"', skipinitialspace=True)
        
        for line in csvreader:
            if first_line:
                array_size = len(line)
                first_line = False
            else:
                data = []
                for i in range(array_size):
                    data.append(float(line[i]))
                    
                track_data.append(data)

    # add "NormalizedSplinePosition"
    #['Position_X', 'Position_Y', 'Position_Z', 'NormalizedSplinePosition', 'Length', 'SideLeft', 'SideRight', 'Radius', 'Speed', 'Brake', 'Camber',  
    # 'Direction', 'Grade', 'Normal_X', 'Normal_Y', 'Normal_Z', 'Forward_X', 'Forward_Y', 'Forward_Z', 'Tag', 'LagG_Obsolete']
    
    track_data = np.array(track_data)
    track_length = track_data[-1,3]
    
    if track_length <= 0.0:
        track_length = 1.0
    
    nsp_column = track_data[:, 3] / track_length
    track_data = np.insert(track_data, 3, nsp_column, axis=1)
    
    #compute and SideLeft XYZ and SideRight XYZ
    centerline_xyz = track_data[:, 0:3]
    normal_xyz =  track_data[:, 13:16]
    forward_xyz = track_data[:, 16:19]
    side_dir = np.cross(normal_xyz, forward_xyz)
    norms = np.linalg.norm(side_dir, axis=1)[np.newaxis] 
    normalized_side_dir = side_dir / norms.T
    '''
    print (centerline_xyz)
    print("----------")
    print (normal_xyz)
    print("----------")
    print (forward_xyz)
    print("----------")
    print (side_dir)
    print("----------")
    print (normalized_side_dir)
    '''
    
    sideleft_xyz = centerline_xyz + normalized_side_dir * track_data[:, 5][np.newaxis].T
    sideright_xyz = centerline_xyz + normalized_side_dir * -track_data[:, 6][np.newaxis].T
    
    # DEBUG : for now, use the AI line forward direction
    direction_xyz = np.array(forward_xyz)
    
    # project 3D coordinates arrays into 2D coordinates
    centerline_xy = np.delete(centerline_xyz, obj=1, axis=1)
    centerline_xy[:,1] *= -1
    sideleft_xy = np.delete(sideleft_xyz, obj=1, axis=1)
    sideleft_xy[:,1] *= -1
    sideright_xy = np.delete(sideright_xyz, obj=1, axis=1)
    sideright_xy[:,1] *= -1
    direction_xy = np.delete(direction_xyz, obj=1, axis=1)
    direction_xy[:,1] *= -1
    
    
# DEBUG !!!!!!!!!!!!!!!!

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot()

tr = mtransforms.Affine2D().rotate_deg(CIRCUIT_ROTATION).translate(CIRCUIT_TRANSLATE_X, CIRCUIT_TRANSLATE_Y) + ax.transData

ax.scatter(centerline_xy[:,0], centerline_xy[:,1], c="g", marker="o", s=0.2*(72./fig.dpi)**2, transform=tr)
ax.scatter(sideleft_xy[:,0], sideleft_xy[:,1], c="r", marker="o", s=0.2*(72./fig.dpi)**2, transform=tr)
ax.scatter(sideright_xy[:,0], sideright_xy[:,1], c="b", marker="o", s=0.2*(72./fig.dpi)**2, transform=tr)

# END DEBUG !!!!!!!!!!!!!!!!


for pattern in FILE_FILTER:
    filenames = glob.glob(input_path + '*' + pattern + '*.txt')
    
    for filename in filenames:
        #print(filename)

        trajectory = []
        first_line = True
        idx = [0] * 23
        
        with open(filename, newline='') as csvfile:
            csvreader = reader(csvfile, delimiter=',', quotechar='"', skipinitialspace=True)
            
            for line in csvreader:
                if first_line:
                    idx[0] = line.index("WorldPosition_X")
                    idx[1] = line.index("WorldPosition_Y")
                    idx[2] = line.index("WorldPosition_Z")
                    idx[3] = line.index("NormalizedSplinePosition")
                    idx[4] = line.index("LapTime")
                    idx[5] = line.index("Gas")
                    idx[6] = line.index("Brake")
                    idx[7] = line.index("LocalVelocity_X")
                    idx[8] = line.index("LocalVelocity_Y")
                    idx[9] = line.index("LocalVelocity_Z")
                    idx[10] = line.index("AccG_X")
                    idx[11] = line.index("AccG_Z")
                    idx[12] = line.index("Velocity_X")
                    idx[13] = line.index("Velocity_Y")
                    idx[14] = line.index("Velocity_Z")
                    idx[15] = line.index("SlipAngle_W")
                    idx[16] = line.index("SlipAngle_X")
                    idx[17] = line.index("SlipAngle_Y")
                    idx[18] = line.index("SlipAngle_Z")
                    idx[19] = line.index("SlipRatio_W")
                    idx[20] = line.index("SlipRatio_X")
                    idx[21] = line.index("SlipRatio_Y")
                    idx[22] = line.index("SlipRatio_Z")
                    
                    first_line = False
                else:
                    velocity = [float(line[idx[7]]), float(line[idx[8]]), float(line[idx[9]])]
                    velocity_norm = float(np.sqrt(np.sum(np.array(velocity)**2)))
                    
                    #  0:WPosX        1:WPosY,       2:WPosZ        3:TrackPos
                    #  4:LapT         5:Trottle      6:Brake        7:Veloc  
                    #  8:AccLat       9:AccLong     10:VelocX      11:VelocY
                    # 12:VelocZ      13:SlipAngleW  14:SlipAngleX  15:SlipAngleY
                    # 16:SlipAngleZ  17:SlipRatioW  18:SlipRatioX  19:SlipRatioY
                    # 20:SlipRatioZ
                    
                    data = [float(line[idx[0]]), float(line[idx[1]]), float(line[idx[2]]), float(line[idx[3]]),
                            float(line[idx[4]]), float(line[idx[5]]), float(line[idx[6]]), velocity_norm, 
                            float(line[idx[10]]), float(line[idx[11]]), float(line[idx[12]]), float(line[idx[13]]), float(line[idx[14]]),
                            float(line[idx[15]]), float(line[idx[16]]), float(line[idx[17]]), float(line[idx[18]]), 
                            float(line[idx[19]]), float(line[idx[20]]), float(line[idx[21]]), float(line[idx[22]])]
                    trajectory.append(data)
        filename_array.append(filename)

        trajectory = np.array(trajectory)
        trajectories.append(trajectory)
        

        
#extract lap times
laps = np.zeros((len(trajectories), 3)).astype(np.int32)

best_lap = 3600000
worst_lap = 0
best_lap_id = -1
worst_lap_id = -1
lap_delta = 0

for i in range(len(trajectories)):
    lap_valid = True
    traj = trajectories[i]
    
    # check if lap is starting and ending at correct locations
    if traj[0,3] > 0.003 or (traj[-1,3] < 0.997 and traj[-2,3] < 0.997):
        lap_valid = False
    else:
        # check if teleport occured between starting and ending of lap
        spos_diff = np.diff(traj[:,3])
        reset_id = np.argwhere(spos_diff < 0)
        if reset_id.size > 0:
            if np.amin(reset_id) != spos_diff.shape[0]-1:
                lap_valid = False

    if lap_valid:
        lap_time_ms = np.rint(traj[-1,4]).astype(np.int32)    # TODO:  improve extact lap time computation, based on spline pos ratio
        laps[i,:] = [i, lap_time_ms, 0]

        if lap_time_ms < best_lap:
            best_lap = lap_time_ms
            best_lap_id = i
        if lap_time_ms > worst_lap:
            worst_lap = lap_time_ms
            worst_lap_id = i 
    else:
        laps[i,:] = [i, 0, 0]

if best_lap_id >= 0:
    laps[best_lap_id, 2] = 1
if worst_lap_id >= 0:
    laps[worst_lap_id, 2] = -1
    if best_lap_id >= 0:
        lap_delta = worst_lap - best_lap


# filter out lap times with large delta compared to best lap
if DISPLAY_ONLY_LAP_DELTA_MAX > 0:
    worst_lap = 0
    worst_lap_id = -1

    for i in range(laps.shape[0]):
        if laps[i,1] > best_lap + 1000 * DISPLAY_ONLY_LAP_DELTA_MAX:
            laps[i,1] = 0
            laps[i,2] = 0
            
    for i in range(laps.shape[0]):
        if laps[i,1] > worst_lap:
            worst_lap = laps[i,1]
            worst_lap_id = i 
    
    if worst_lap_id >= 0:
        laps[worst_lap_id, 2] = -1
        if best_lap_id >= 0:
            lap_delta = worst_lap - best_lap

# display lap times
if DISPLAY_LAP_TIME:
    for i in range(laps.shape[0]):
        lap_time_ms = laps[i,1]
        if lap_time_ms > 0:
            lap_time_sec = 0.001*lap_time_ms
            lap_min = np.floor(lap_time_sec).astype(np.int32) // 60
            lap_sec = np.floor(lap_time_sec - 60*lap_min).astype(np.int32)
            lap_msec = np.rint(lap_time_ms - 1000*(60*lap_min + lap_sec)).astype(np.int32)

            if laps[i,2] == 1:
                print("Lap Time: %d:%02d.%03d  <==  Best lap time :   %s" % (lap_min, lap_sec, lap_msec, os.path.basename(filename_array[i])))
            elif laps[i,2] == -1:
                print("Lap Time: %d:%02d.%03d  <==  Worste lap time : %s" % (lap_min, lap_sec, lap_msec, os.path.basename(filename_array[i])))
            else:
                print("Lap Time: %d:%02d.%03d              lap time : %s" % (lap_min, lap_sec, lap_msec, os.path.basename(filename_array[i])))


# compile the list of valid laps
laps2 = []
interp_kind = 'linear' #'quadratic'
for i in range(len(trajectories)):
    if not DISPLAY_ONLY_FULL_LAPS or laps[i,1] > 0:
        laps2.append(laps[i])
        traj = trajectories[i]
        valid_trajectories.append(traj)
        
        # apply linear interpolation to align data with track position (tp) instead of time
        tp = traj[:,3]   # trackpos in %
        
        # correct last 5 tp data if below 50%  (correction for tp rolled back to zero)
        tp_tmp = tp[-5:]
        tp_tmp = np.where(tp_tmp < 0.5, tp_tmp + 1.0, tp_tmp)
        tp[-5:] = tp_tmp
        tp_tick = 0.0005   # 0.05%   =>  2000 data point per lap  ~=  every 2.3 m in Barcelona
        tp2 = np.arange(0.0, 1.0 + tp_tick, tp_tick)
        
        tp_traj = []
        
        #print(traj.shape, traj.shape[1])
        
        for i in range(traj.shape[1]):
            y = traj[:,i]
            f = interp1d(tp, y, kind=interp_kind, bounds_error=False, fill_value="extrapolate")
            y2 = f(tp2)
            tp_traj.append(y2)
        
        tp_traj = np.array(tp_traj).transpose()
        valid_tp_trajectories.append(tp_traj)
        
laps2 = np.array(laps2)
    
best_valid_lap_id = np.argwhere(laps2[:,2] == 1).item()



# lidar 2D
# interesctions based on https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

def segment_length(px, py, qx, qy):
    sqlen = (px - qx)*(px - qx) + (py - qy)*(py - qy)
    return math.sqrt(sqlen)

    
def on_segment(px, py, qx, qy, rx, ry):
    
    if ( (qx <= max(px, rx)) and (qx >= min(px, rx)) and
           (qy <= max(py, ry)) and (qy >= min(py, ry))):
        return True
        
    return False


def orientation(px, py, qx, qy, rx, ry):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
     
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ for details of below formula.
     
    val = (float(qy - py) * (rx - qx)) - (float(qx - px) * (ry - qy))
    if (val > 0):    # Clockwise orientation
        return 1
    elif (val < 0):  # Counterclockwise orientation
        return 2
    else:            # Collinear orientation
        return 0


def orientation_vec(p_array, q_array, r_array):
    
    val = np.multiply(q_array[:,1] - p_array[:,1], r_array[:,0] - q_array[:,0]) - np.multiply(q_array[:,0] - p_array[:,0], r_array[:,1] - q_array[:,1])
    val = np.where(val > 0, 1, val)  # Clockwise orientation
    val = np.where(val < 0, 2, val)  # Counterclockwise orientation
                                     # Collinear orientation when  val[:] == 0
    
    return val.astype(np.int32)


def do_intersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
     
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1x, p1y, q1x, q1y, p2x, p2y)
    o2 = orientation(p1x, p1y, q1x, q1y, q2x, q2y)
    o3 = orientation(p2x, p2y, q2x, q2y, p1x, p1y)
    o4 = orientation(p2x, p2y, q2x, q2y, q1x, q1y)
 
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
 
    # Special Cases
 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and on_segment(p1x, p1y, p2x, p2y, q1x, q1y)):
        return True
 
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and on_segment(p1x, p1y, q2x, q2y, q1x, q1y)):
        return True
 
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and on_segment(p2x, p2y, p1x, p1y, q2x, q2y)):
        return True
 
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and on_segment(p2x, p2y, q1x, q1y, q2x, q2y)):
        return True
 
    # If none of the cases
    return False
 
 
def do_intersect_vec(p1_array, q1_array, p2_array, q2_array):
    
    
    # Find the 4 orientations required for the general and special cases
    o1 = orientation_vec(p1_array, q1_array, p2_array)
    o2 = orientation_vec(p1_array, q1_array, q2_array)
    o3 = orientation_vec(p2_array, q2_array, p1_array)
    o4 = orientation_vec(p2_array, q2_array, q1_array)
    
    val = np.logical_and(np.not_equal(o1,o2), np.not_equal(o3,o4))
    
    # TODO manage special cases !!!!!!!!!!!!!!!!!!!!
    
    '''
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
 
    # Special Cases
 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and on_segment(p1x, p1y, p2x, p2y, q1x, q1y)):
        return True
 
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and on_segment(p1x, p1y, q2x, q2y, q1x, q1y)):
        return True
 
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and on_segment(p2x, p2y, p1x, p1y, q2x, q2y)):
        return True
 
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and on_segment(p2x, p2y, q1x, q1y, q2x, q2y)):
        return True
 
    # If none of the cases
    return False
    '''
    
    return val
    

def compute_intersection(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
    #compute intesection, see  https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
        
    # u = (q − p) × r / (r × s)         with p1 = p    q1 = p+r   p2 = q   q2 = q+s      r = q1 - p1       s = q2 - p2
    # u = (p2 - p1) × (q1 - p1) / ((q1 - p1) × (q2 - p2))
    
    v1x = p2x - p1x
    v1y = p2y - p1y
    v2x = q1x - p1x
    v2y = q1y - p1y
    v3x = q2x - p2x
    v3y = q2y - p2y
    
    v1xv2 = v1x*v2y - v1y*v2x
    v2xv3 = v2x*v3y - v2y*v3x
    
    u = v1xv2 / v2xv3
    
    ix = p2x + u*v3x
    iy = p2y + u*v3y
    return ix, iy


# TODO vecotrize the list of lidar angles
def compute_lidar(max_dist, half_angle, angle_inc, dir_xy, p1_array, p2_array, q2_array, p3_array, q3_array):
    
    lidar_segments = []
    
    p1x = p1_array[0,0]
    p1y = p1_array[0,1]

    for dir_angle in np.arange(-half_angle, half_angle + angle_inc, angle_inc): 
        dir_angle_rad = math.radians(dir_angle)
        
        ccos = math.cos(dir_angle_rad)
        csin = math.sin(dir_angle_rad)
        #print(dir_angle, dir_angle_rad, ccos, csin)
        
        current_dir_x = ccos*dir_xy[0] - csin*dir_xy[1]
        current_dir_y = csin*dir_xy[0] + ccos*dir_xy[1]
        
        
        q1x = p1x + max_dist*current_dir_x       # 2km forward looking
        q1y = p1y + max_dist*current_dir_y

        q1_array = [q1x, q1y]
        q1_array = np.array(q1_array)[np.newaxis].astype(intersect_type)
        q1_array = np.repeat(q1_array, repeats=size, axis=0)
        
        line_segments = []
        shortest_length = max_dist
        
        # check intersetion on track side left
        val = do_intersect_vec(p1_array, q1_array, p2_array, q2_array)
        inter_id = np.argwhere(val==True).flatten()
        
        for j in range(inter_id.shape[0]):
            k = inter_id[j]
            ix, iy = compute_intersection(p1_array[k,0], p1_array[k,1], q1_array[k,0], q1_array[k,1], 
                                          p2_array[k,0], p2_array[k,1], q2_array[k,0], q2_array[k,1])
        
            clen = segment_length(p1x, p1y, ix, iy)
            if clen < shortest_length:
                line_segments.append([(p1x, p1y), (ix, iy)])
                shortest_length = clen
        
        # check intersetion on track side right
        val = do_intersect_vec(p1_array, q1_array, p3_array, q3_array)
        inter_id = np.argwhere(val==True).flatten()
        
        for j in range(inter_id.shape[0]):
            k = inter_id[j]
            ix, iy = compute_intersection(p1_array[k,0], p1_array[k,1], q1_array[k,0], q1_array[k,1], 
                                          p3_array[k,0], p3_array[k,1], q3_array[k,0], q3_array[k,1])

            clen = segment_length(p1x, p1y, ix, iy)
            if clen < shortest_length:
                line_segments.append([(p1x, p1y), (ix, iy)])
                shortest_length = clen
        
                 
        line_segments = np.array(line_segments)
        
        if line_segments.shape[0] > 0:
            lidar_segments.append(line_segments[-1,:,:])
            

    lidar_segments = np.array(lidar_segments)   
    return lidar_segments




# DEBUG !!!!!!!!!!!!!!!!


first_traj = valid_trajectories[0]

ax.plot(first_traj[:,0], -first_traj[:,2], c="k", linewidth=1.0, alpha=0.7, transform=tr)
ax.plot(centerline_xy[:,0], centerline_xy[:,1], c="g", linewidth=0.4, alpha=0.4, transform=tr)
ax.plot(sideleft_xy[:,0], sideleft_xy[:,1], c="r", linewidth=0.4, alpha=0.4, transform=tr)
ax.plot(sideright_xy[:,0], sideright_xy[:,1], c="b", linewidth=0.4, alpha=0.4, transform=tr)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')


# compute 2D lidar

max_dist = 2000  # 2km forward looking
half_angle = 60
angle_inc = 2

#id = 2680
id = 520

size = sideleft_xy.shape[0]
#p1x = centerline_xy[id,0]
#p1y = centerline_xy[id,1]

intersect_type = np.single

# vectorize intersection computation
p1_array = [centerline_xy[id,0], centerline_xy[id,1]]
p1_array = np.array(p1_array)[np.newaxis].astype(intersect_type)
p1_array = np.repeat(p1_array, repeats=size, axis=0)

p2_array = sideleft_xy.astype(intersect_type)
q2_array = np.roll(p2_array,-1, axis=0)

p3_array = sideright_xy.astype(intersect_type)
q3_array = np.roll(p3_array,-1, axis=0)


dir_xy = [direction_xyz[id, 0], -direction_xyz[id, 2]]


start = time.time()
lidar_segments = compute_lidar(max_dist, half_angle, angle_inc, dir_xy, p1_array, p2_array, q2_array, p3_array, q3_array)
end = time.time()
print("intersection execution: %.2f ms" % (1000.0*(end - start)))


#print(lidar_segments.shape)

if lidar_segments.shape[0] > 0:
    for k in range(lidar_segments.shape[0]):
        ax.plot(lidar_segments[k,:,0], lidar_segments[k,:,1], c="m", linewidth=0.5, alpha=0.5, transform=tr)


ax.set_xlim(-CIRCUIT_LIMIT_X, CIRCUIT_LIMIT_X)
ax.set_ylim(-CIRCUIT_LIMIT_Y, CIRCUIT_LIMIT_Y)
ax.set_aspect('equal')

plt.show()

#exit()

# END DEBUG !!!!!!!!!!!!!!!!



    
def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    hsv=np.empty_like(rgb)
    hsv[...,3:]=rgb[...,3:]
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    maxc = np.max(rgb[...,:2],axis=-1)
    minc = np.min(rgb[...,:2],axis=-1)    
    hsv[...,2] = maxc   
    hsv[...,1] = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    hsv[...,0] = np.select([r==maxc,g==maxc],[bc-gc,2.0+rc-bc],default=4.0+gc-rc)
    hsv[...,0] = (hsv[...,0]/6.0) % 1.0
    idx=(minc == maxc)
    hsv[...,0][idx]=0.0
    hsv[...,1][idx]=0.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    rgb=np.empty_like(hsv)
    rgb[...,3:]=hsv[...,3:]    
    h,s,v=hsv[...,0],hsv[...,1],hsv[...,2]   
    i = (h*6.0).astype('uint8')
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    conditions=[s==0.0,i==1,i==2,i==3,i==4,i==5]
    rgb[...,0]=np.select(conditions,[v,q,p,p,t,v],default=v)
    rgb[...,1]=np.select(conditions,[v,v,v,q,p,p],default=t)
    rgb[...,2]=np.select(conditions,[v,p,t,v,v,q],default=p) 
    return rgb
    
    
class Plotter:
    def __init__(self, pax):
        self.ax = pax
        self.fig = pax.figure
        self.ln = []
        self.dot = []
        self.lidar = np.zeros([61, 2,2])   # TODO
        self.lidar[:,1,0] = 0.1
        self.lidar[:,1,1] = 0.1
        self.lidar_ln = []
        self.plotax = None
        self.subln = []
        self.subdot = []
        self.obj = None

    def plot(self, obj):
        self.obj = obj
        
        tr = self.obj.transform
        if tr is None:
            tr = mtransforms.Affine2D.identity()
        
        nb_series = obj.get_series_size()
        
        if obj.mode == 'classic':
            for i in range(nb_series):
                ln = self.ax.plot(obj.xseries[i], obj.yseries[i], color=obj.colors[i], linewidth=obj.line_width, alpha=obj.line_alpha, visible=obj.visible[i], label=obj.labels[i])
                c = obj.colors[i]
                dot = self.ax.plot(obj.xseries[i][0], obj.yseries[i][0], color=(c[0],c[1],c[2],0.4), marker='o', markersize=obj.marker_size, visible=obj.visible[i], label=obj.labels[i])
                self.ln.append(ln)
                self.dot.append(dot)
        elif obj.mode == 'polar':
            self.ax.axis('off')
            polar_ax = plt.axes([0.2, 0.1, 0.6, 0.6], projection='polar')
            self.plotax = polar_ax
            self.update_polar_plot(self.obj.history)
        elif obj.mode == 'track':
            self.ax.axis('off')
            self.plotax = self.ax
            for i in range(nb_series):            
                ln = self.ax.plot(obj.xseries[i], obj.yseries[i], color=obj.colors[i], linewidth=obj.line_width, alpha=obj.line_alpha, visible=obj.visible[i], label=obj.labels[i], transform=tr)
                c = obj.colors[i]
                dot = self.ax.plot(obj.xseries[i][0], obj.yseries[i][0], color=(c[0],c[1],c[2],0.4), marker='o', markersize=obj.marker_size, visible=obj.visible[i], label=obj.labels[i], transform=tr)
                
                lidar_lines = []
                lidar_size = self.lidar.shape[0]                                    
                for k in range(lidar_size):
                    lidar_line = self.ax.plot(self.lidar[k,:,0], self.lidar[k,:,1], c="m", linewidth=0.5, alpha=0.5, visible=obj.visible[i], label=obj.labels[i], transform=tr)
                    lidar_lines.append(lidar_line)
                
                self.ln.append(ln)
                self.dot.append(dot)
                self.lidar_ln.append(lidar_lines)
                
            #self.ax.set_xlim(-CIRCUIT_LIMIT_X, CIRCUIT_LIMIT_X)
            #self.ax.set_ylim(-CIRCUIT_LIMIT_Y, CIRCUIT_LIMIT_Y)
            #self.ax.set_aspect('equal')
        
        if obj.ax_inset is not None:
            nb_subseries = obj.get_subseries_size()
            for i in range(nb_subseries):
                ln = obj.ax_inset.plot(obj.xsubseries[i], obj.ysubseries[i], color=obj.subcolors[i], linewidth=1.0, alpha=1.0, transform=obj.inset_tr)
                dot = obj.ax_inset.plot(obj.xsubseries[i][0], obj.ysubseries[i][0], color=obj.subcolors[i], marker='o', alpha=1.0, markersize=7, transform=obj.inset_tr)
                self.subln.append(ln)
                self.subdot.append(dot)
            
        _vars = obj.get_variables()
        plt.subplots_adjust(bottom=0.03*(len(_vars)+2))
        self.sliders = []
        for i,var in enumerate(_vars):
            self.add_slider(i*0.03, var[0], var[1], var[2], var[3])
        
        #self.fig.canvas.draw_idle()

    def draw(self):
        self.ax.draw()

    def get_plot(self):
        return self.obj

    def clear(self):
        if self.obj is not None:
            del self.obj
            self.obj = None
        self.ln = []
        self.dot = []
    
    def update_polar_plot(self, val):
        self.plotax.clear()
        self.ln = []
        self.dot = []
        
        if self.obj.maxz <= 2.0:
            tick_array = [0.5, 1, 1.5, 2]
            self.plotax.set_ylim(0.0 ,2.0)
        else:
            tick_count = np.floor(2*self.obj.maxz).astype(np.int32) + 1
            tick_array = np.linspace(0.0, 0.5*tick_count, tick_count+1)
            self.plotax.set_ylim(0.0, 0.5*tick_count)
            
        self.plotax.set_rticks(tick_array)                    
        self.plotax.set_rgrids(tick_array, angle=30.)
        self.plotax.xaxis.grid(True,color=(0.9,0.9,0.9),linestyle='-')
        self.plotax.yaxis.grid(True,color=(0.9,0.9,0.9),linestyle='-')
        
        for i in range(self.obj.get_series_size()):
            x, y = self.obj.get_series(i)
            pos_cur = val
            pos_prev = np.clip(pos_cur - self.obj.history, 0, 100)
            idx_cur = (np.abs(self.obj.xseries[i] - pos_cur)).argmin()
            idx_prev = (np.abs(self.obj.xseries[i] - pos_prev)).argmin()
            series_length = len(self.obj.xseries[i])
            c = self.obj.colors[i]
            r, g, b = to_rgb(self.obj.colors[i])
            alpha_arr = np.zeros(series_length)
            array = np.linspace(0.0, self.obj.marker_alpha, idx_cur+1 - idx_prev)
            alpha_arr[idx_prev:idx_cur+1] = array
            color_array = [(r, g, b, alpha) for alpha in alpha_arr]
            array = np.linspace(1.0, self.obj.marker_size, idx_cur+1 - idx_prev)
            size_arr = np.zeros(series_length)
            size_arr[idx_prev:idx_cur+1] = array
            ln = self.plotax.plot(self.obj.yseries[i][idx_prev:idx_cur+1], self.obj.zseries[i][idx_prev:idx_cur+1], color=c, linewidth=self.obj.line_width, alpha=self.obj.line_alpha, visible=self.obj.visible[i], label=self.obj.labels[i])
            dot = self.plotax.scatter(self.obj.yseries[i], self.obj.zseries[i], c=color_array, marker='o', s=size_arr, visible=self.obj.visible[i], label=self.obj.labels[i])
            self.ln.append(ln)
            self.dot.append(dot)

    def add_slider(self, pos, name, varname, min, max):
        sax = plt.axes([0.2, 0.01+pos, 0.7, 0.02], facecolor='lightgoldenrodyellow')
        slider = Slider(sax, name, min, max, valinit=getattr(self.obj, varname))
        self.sliders.append(slider)
        
        def update(val):
            setattr(self.obj, varname, val)
            idx = 0
            if self.obj.mode == 'classic':
                for i in range(self.obj.get_series_size()):
                    if self.obj.visible[i]:
                        x, y = self.obj.get_series(i)
                        idx = (np.abs(x - val)).argmin()
                        self.dot[i][0].set_xdata(x[idx])
                        self.dot[i][0].set_ydata(y[idx])
                        
            elif self.obj.mode == 'polar':
                self.update_polar_plot(val)
                # self.obj.trackpos    instead of val ??
                
            elif self.obj.mode == 'track':
                self.ax.set_aspect('equal')
                once = True
                for i in range(self.obj.get_series_size()):
                    if self.obj.visible[i]:
                        x, y = self.obj.get_series(i)
                        idx = np.ceil(0.01 * x.shape[0] * self.obj.trackpos).astype(int)
                        self.dot[i][0].set_xdata(x[idx])
                        self.dot[i][0].set_ydata(y[idx])
                        
                        # compute lidar
                        if once:
                            # TODO
                            max_dist = 2000  # 2km forward looking
                            half_angle = 60
                            angle_inc = 2
    
                            size = sideleft_xy.shape[0]
                            
                            intersect_type = np.single
                            p1_array = [x[idx], y[idx]]
                            p1_array = np.array(p1_array)[np.newaxis].astype(intersect_type)
                            p1_array = np.repeat(p1_array, repeats=size, axis=0)

                            p2_array = sideleft_xy.astype(intersect_type)
                            q2_array = np.roll(p2_array,-1, axis=0)

                            p3_array = sideright_xy.astype(intersect_type)
                            q3_array = np.roll(p3_array,-1, axis=0)

                            id1 = (idx+1) % x.shape[0]
                            id2 = (idx-1) % x.shape[0]
                            dir_xy = [x[id1] - x[id2], y[id1] - y[id2]]

                            start = time.time()
                            lidar_segments = compute_lidar(max_dist, half_angle, angle_inc, dir_xy, p1_array, p2_array, q2_array, p3_array, q3_array)
                            end = time.time()
                            print("compute lidar execution: %.2f ms" % (1000.0*(end - start)))
                            once = False
                            self.lidar = lidar_segments
                            
                            for k in range(lidar_segments.shape[0]): 
                                self.lidar_ln[i][k][0].set_data(lidar_segments[k,:,0], lidar_segments[k,:,1])
                                
            
            
            if self.obj.ax_inset is not None:
                nb_subseries = self.obj.get_subseries_size()
                for i in range(nb_subseries):
                    x2, y2 = self.obj.get_subseries(i)
                    idx2 = np.ceil(0.01 * x2.shape[0] * self.obj.trackpos).astype(int)
                    self.subdot[i][0].set_xdata(x2[idx2])
                    self.subdot[i][0].set_ydata(y2[idx2])
                                
                
            self.fig.canvas.draw_idle()
        slider.on_changed(update)

class TrackProgressionPlot:
    def __init__(self, mode=None):
        self.mode = mode
        self.ax = None
        self.trackpos = 0.0
        self.xseries = []
        self.yseries = []
        self.zseries = []
        self.colors = []
        self.visible = []
        self.labels = []
        self.line_width = 1.0
        self.line_alpha = 1.0
        self.line_color = 'b'
        self.marker_size = 5.0
        self.marker_alpha = 1.0
        self.history = 0.0
        self.ax_inset = None
        self.inset_tr = None
        self.xsubseries = []
        self.ysubseries = []
        self.subcolors = []
        self.varables = [('Track Position', 'trackpos', 0, 100)]
        self.minx = 100000
        self.miny = 100000
        self.minz = 100000
        self.maxx = -100000
        self.maxy = -100000
        self.maxz = -100000
        self.transform = None

    def set_params(self, mode, line_width, line_alpha, line_color, marker_size, marker_alpha, transform=None):
        self.mode = mode
        self.line_width = line_width
        self.line_alpha = line_alpha
        self.line_color = line_color
        self.marker_size = marker_size
        self.marker_alpha = marker_alpha
        self.transform = transform

    def set_history(self, history):
        self.history = history
    
    def set_axis(self, ax):
        self.ax = ax
        print (type(ax))
    
    def add_series(self, x, y, z, col, vis, label):
        self.xseries.append(x)
        self.yseries.append(y)
        self.zseries.append(z)
        self.colors.append(col)
        self.visible.append(vis)
        self.labels.append(label)
        
        if x is not None:
            if (np.amin(x) < self.minx):
                self.minx = np.amin(x)
            if (np.amax(x) > self.maxx):
                self.maxx = np.amax(x)
        if y is not None:
            if (np.amin(y) < self.miny):
                self.miny = np.amin(y)
            if (np.amax(y) > self.maxy):
                self.maxy = np.amax(y)
        if z is not None:
            if (np.amin(z) < self.minz):
                self.minz = np.amin(z)
            if (np.amax(z) > self.maxz):
                self.maxz = np.amax(z)
        
    def add_inset(self, ax, tr):
        self.ax_inset = ax
        self.inset_tr = tr
    
    def add_subseries(self, x, y, col):
        if self.ax_inset is not None:
            self.xsubseries.append(x)
            self.ysubseries.append(y)
            self.subcolors.append(col)
        
    def get_series(self, i=0):
        if i < len(self.xseries):
            return self.xseries[i], self.yseries[i]
        else:
            return None
            
    def get_subseries(self, i=0):
        if i < len(self.xsubseries):
            return self.xsubseries[i], self.ysubseries[i]
        else:
            return None

    def get_series_size(self):
        return len(self.xseries)
        
    def get_subseries_size(self):
        return len(self.xsubseries)

    def get_variables(self):
        return self.varables
        




def redraw_plot(fig, display_mode, radio):
        
    for i in range(len(fig.axes)-1, -1, -1):
        if i >= 2:
            fig.axes[i].remove()
        
    ax = fig.add_subplot(gs[1:, 1:], label=display_mode)
    config_plot(ax, display_mode)
    
    #artists = fig.axes[0].get_children()
    #for obj in artists:
    #    print(type(obj), obj.get_visible())
    
    #fig.axes[0].clear()
    #check = config_checkbuttons(fig)
    
    #artists = fig.axes[0].get_children()
    #for obj in artists:
    #    print(type(obj), obj.get_visible())
    
    #radio.set_checkbuttons(check)
    plt.draw()


class MyRadioButtons(RadioButtons):
    
    def __init__(self, fig, ax, labels, active=0, activecolor='blue', size=49,
                 orientation="vertical", **kwargs):
        """
        Add radio buttons to an `~.axes.Axes`.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button.
        size : float
            Size of the radio buttons
        orientation : str
            The orientation of the buttons: 'vertical' (default), or 'horizontal'.
        Further parameters are passed on to `Legend`.
        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        axcolor = ax.get_facecolor()
        self.value_selected = None
        self.checkbuttons = None
        self.fig = fig

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor
            p = ax.scatter([],[], s=size, marker="o", edgecolor='black',
                           facecolor=facecolor)
            circles.append(p)
        if orientation == "horizontal":
            kwargs.update(ncol=len(labels), mode="expand")
        kwargs.setdefault("frameon", False)    
        self.box = ax.legend(circles, labels, loc="center", **kwargs)
        self.labels = self.box.texts
        self.circles = self.box.legendHandles
        for c in self.circles:
            c.set_picker(8)
        self.cnt = 0
        self.observers = {}

        self.connect_event('pick_event', self._clicked)


    def _clicked(self, event):
        if (self.ignore(event) or event.mouseevent.button != 1 or
            event.mouseevent.inaxes != self.ax):
            return
        if event.artist in self.circles:
            index = self.circles.index(event.artist)
            self.set_active(index)
            #print("clicked :", index, self.value_selected)
            
            redraw_plot(self.fig, index, self)
            
    def set_checkbuttons(self, check):
        self.checkbuttons = check




def config_plot(ax, display_mode):
    global laplines, checkbtn, plotter, LINE_WIDTH, LINE_ALPHA, LINE_COLOR
    
    if 'plotter' in globals():
        del plotter
        
    plotter = Plotter(ax)
    
    init_state = False
    laplines.clear()
    status = [False] * len(valid_trajectories)
    min_veloc = 1000
    max_veloc = 0
    min_curv = 200
    max_curv = -200
    velocities = np.array([max_veloc, min_veloc])
    curvatures = np.array([min_curv, max_curv])
    
    if display_mode in [DISPLAY_MODE_VELOCITY_PLOT, DISPLAY_MODE_G_PLOT, DISPLAY_MODE_TRAJ_DIST, DISPLAY_MODE_VELOC_ANGLE, DISPLAY_MODE_TIRE_SLIP_ANGLE, DISPLAY_MODE_TIRE_SLIP_RATIO, DISPLAY_MODE_LIDAR]:
        traj_array = np.concatenate(valid_trajectories, axis=0)
        ax.set_xlim(0.0, 100.0)
        
        if display_mode == DISPLAY_MODE_VELOCITY_PLOT:
            ax.set_ylim(0.9*3.6*np.amin(traj_array[:,7]), 1.1*3.6*np.amax(traj_array[:,7]))
        elif display_mode == DISPLAY_MODE_TRAJ_DIST:
            ax.set_ylim(-10, 10)
        elif display_mode == DISPLAY_MODE_VELOC_ANGLE:
            ax.set_ylim(-15, 15)
        elif display_mode == DISPLAY_MODE_TIRE_SLIP_ANGLE:
            ax.set_ylim(-25, 25)
        elif display_mode == DISPLAY_MODE_TIRE_SLIP_RATIO:
            ax.set_ylim(-1, 1)
        elif display_mode == DISPLAY_MODE_LIDAR:
            ax.set_xlim(-CIRCUIT_LIMIT_X, CIRCUIT_LIMIT_X)
            ax.set_ylim(-CIRCUIT_LIMIT_Y, CIRCUIT_LIMIT_Y)
            ax.set_aspect('equal')
            
        ax.set_aspect('auto')
        ax.grid(b=True, which='both', axis='y') 
        plt.subplots_adjust(bottom=0.1, top=0.95)
        
        # plot inset to display the track
        ax_inset = ax.inset_axes([0.75, 0.75, 0.25, 0.25])
        ax_inset.spines['left'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['bottom'].set_visible(False)
        ax_inset.set_facecolor((1, 1, 1, 0.7))
        ax_inset.get_xaxis().set_visible(False)
        ax_inset.get_yaxis().set_visible(False)
        ax_inset.set_xlim(-CIRCUIT_LIMIT_X, CIRCUIT_LIMIT_X)
        ax_inset.set_ylim(-CIRCUIT_LIMIT_Y, CIRCUIT_LIMIT_Y)
        ax_inset.set_aspect('equal')
        
    else:
        ax.set_xlim(-CIRCUIT_LIMIT_X, CIRCUIT_LIMIT_X)
        ax.set_ylim(-CIRCUIT_LIMIT_Y, CIRCUIT_LIMIT_Y)
        ax.set_aspect('equal')
        plt.subplots_adjust(bottom=None, top=None)
    
    if 'checkbtn' in globals():
        status = checkbtn.get_status()
    else:
        init_state = True
        
    # init with global values
    line_width = LINE_WIDTH
    line_alpha = LINE_ALPHA
    line_color = LINE_COLOR 
    
    # display circuit color
    if DISPLAY_ONLY_FULL_LAPS:    
        newcmp = plt.cm.jet(np.arange(plt.cm.jet.N))
        newcmp[:,0:3] *= 0.7   # slightly darker 'jet' colormap
        mycmap = ListedColormap(newcmp)
    
    
    if display_mode == DISPLAY_MODE_PEDALS:    
        col1 = plt.cm.Greens(np.linspace(0., 1, 128))
        col2 = plt.cm.Reds(np.linspace(0., 1, 128))
        # make green look better
        hsv = rgb_to_hsv(col1[:,0:3])
        hsv[:,0] = hsv[:,0] - 0.1
        hsv[:,1] = np.clip(hsv[:,1], 0.2, 1.0)
        hsv[:,2] = np.clip(1.5*hsv[:,2], 0.0, 1.0)
        
        rgb = hsv_to_rgb(hsv)
        col1[:,0:3] = rgb
        col1 = np.flip(col1, 0)   # revert green colormap
        col3 = np.vstack((col1, col2))
        pedal_cmap = ListedColormap(col3)
            
    elif display_mode == DISPLAY_MODE_VELOCITY:
        #newcmp = plt.cm.plasma(np.arange(plt.cm.plasma.N))
        #velocity_cmap = ListedColormap(newcmp)
        newcmp = plt.cm.Spectral(np.arange(plt.cm.Spectral.N))
        newcmp[:,0:3] *= 0.9
        velocity_cmap = ListedColormap(newcmp)
    
    elif display_mode == DISPLAY_MODE_CURVATURE:
        newcmp = plt.cm.bwr(np.arange(plt.cm.bwr.N))
        newcmp[:,0:3] *= 0.85
        curvature_cmap = ListedColormap(newcmp)
        
    elif display_mode == DISPLAY_MODE_VELOCITY_PLOT:
        veloc_plot = TrackProgressionPlot()
        line_width = 1.5
        line_alpha = 0.5
        marker_size = 10.0
        veloc_plot.set_params('classic', line_width, line_alpha, line_color, marker_size, line_alpha)
    
    elif display_mode == DISPLAY_MODE_G_PLOT:
        g_plot = TrackProgressionPlot()
        line_width = 0.6
        line_alpha = 0.2
        marker_size = 100.0
        marker_alpha = 0.6
        g_plot.set_params('polar', line_width, line_alpha, line_color, marker_size, marker_alpha)
        g_plot.set_history(1.5)
    
    elif display_mode == DISPLAY_MODE_TRAJ_DIST:
        traj_disp_plot = TrackProgressionPlot()
        line_width = 3.0
        line_alpha = 0.3
        marker_size = 15.0
        traj_disp_plot.set_params('classic', line_width, line_alpha, line_color, marker_size, line_alpha)
        tp_bl_traj = valid_tp_trajectories[best_valid_lap_id]
        blpx = tp_bl_traj[:,0]      # Best lap world pos X in m    
        blpz = -tp_bl_traj[:,2]     # Best lap world pos Z in m

        blvx = tp_bl_traj[:,10]   # Best lap velocity X in m/s
        blvz = -tp_bl_traj[:,12]  # Best lap velocity Z in m/s
        
        blnormv = np.sqrt(blvx*blvx + blvz*blvz)
        blvdirx = blvx / blnormv
        blvdirz = blvz / blnormv
        
    elif display_mode == DISPLAY_MODE_VELOC_ANGLE:
        veloc_angle_plot = TrackProgressionPlot()
        line_width = 3.0
        line_alpha = 0.3
        marker_size = 15.0
        veloc_angle_plot.set_params('classic', line_width, line_alpha, line_color, marker_size, line_alpha)
        tp_bl_traj = valid_tp_trajectories[best_valid_lap_id]
        blvx = tp_bl_traj[:,10]   # Best lap velocity X in m/s
        blvz = -tp_bl_traj[:,12]  # Best lap velocity Z in m/s
        blv_ang = np.arctan2(blvz, blvx) * 180 / np.pi
        
    elif display_mode == DISPLAY_MODE_TIRE_SLIP_ANGLE:
        slip_angle_plot = TrackProgressionPlot()
        line_width = 0.6
        line_alpha = 0.3
        marker_size = 10.0
        slip_angle_plot.set_params('classic', line_width, line_alpha, line_color, marker_size, line_alpha)
        
    elif display_mode == DISPLAY_MODE_TIRE_SLIP_RATIO:
        slip_ratio_plot = TrackProgressionPlot()
        line_width = 0.6
        line_alpha = 0.3
        marker_size = 10.0
        slip_ratio_plot.set_params('classic', line_width, line_alpha, line_color, marker_size, line_alpha)
        
    elif display_mode == DISPLAY_MODE_LIDAR: 
        lidar_plot = TrackProgressionPlot()
        line_width = 2.0
        line_alpha = 0.5
        marker_size = 8.0
        tr = mtransforms.Affine2D().rotate_deg(CIRCUIT_ROTATION).translate(CIRCUIT_TRANSLATE_X, CIRCUIT_TRANSLATE_Y) + ax.transData
        lidar_plot.set_params('track', line_width, line_alpha, line_color, marker_size, line_alpha, tr)
        
   
    for i in range(len(valid_trajectories)):
        traj = valid_trajectories[i]
        
        if init_state:
            visible = True
        else:
            visible = status[i]
        
        if init_state and HIDE_ALL_NOT_BEST_OR_WORST:
            visible = False
        
        if not DISPLAY_ONLY_FULL_LAPS or laps2[i,1] > 0:
            if DISPLAY_ONLY_FULL_LAPS and lap_delta > 0:
                laptime_ratio = (laps2[i,1] - best_lap) / (lap_delta + 1)       # note: lap_delta + 1 is a fix for the colormap
                line_color = mycmap(laptime_ratio)
                
                if init_state and (display_mode in [DISPLAY_MODE_PEDALS, DISPLAY_MODE_VELOCITY, DISPLAY_MODE_CURVATURE, DISPLAY_MODE_LIDAR]):
                    visible = False
                    
                if display_mode == DISPLAY_MODE_PEDALS:
                    pedal_pos = 0.5*(1.0 - traj[:,5] + traj[:,6])
                    line_width = 10.0
                    line_alpha = 1.0
                elif display_mode == DISPLAY_MODE_VELOCITY:
                    velocities = traj[:,7]
                    line_width = 8.0
                    line_alpha = 1.0
                elif display_mode == DISPLAY_MODE_CURVATURE:
                    # TODO : improvement  resmaple directions along track line (NormalizedSplinePosition)
                    
                    #diffx = np.diff(traj[:,0])
                    #diffy = np.diff(-traj[:,2])
                    param = 5  # ± n datapoint for diff computation
                    diffx = np.roll(traj[:,0], param) - np.roll(traj[:,0], -param)
                    diffy = np.roll(-traj[:,2], param) - np.roll(-traj[:,2], -param)
                    
                    directions = np.arctan2(diffy, diffx)
                    #directions = np.arctan2(np.diff(-traj[:,2]), np.diff(traj[:,0]))
                    #directions = np.append(directions, directions[-1])
                    curvatures = np.diff(directions)
                    curvatures = np.append(curvatures, curvatures[-1])
                    curvatures = np.where(curvatures < 6.0, curvatures, curvatures - 2*np.pi)
                    curvatures = np.where(curvatures > -6.0, curvatures, curvatures + 2*np.pi)
                    line_width = 8.0
                    line_alpha = 1.0
                elif display_mode == DISPLAY_MODE_LIDAR:
                    line_width = 1.0
                    line_alpha = 1.0
                    
            
            x = traj[:,0]
            y = -traj[:,2]
            tr = mtransforms.Affine2D().rotate_deg(CIRCUIT_ROTATION).translate(CIRCUIT_TRANSLATE_X, CIRCUIT_TRANSLATE_Y) + ax.transData
            
            label = "lap %d" % (i+1)
            
            if laps2[i,2] == 1:
                label = label + " B"
                if init_state and HIDE_ALL_NOT_BEST_OR_WORST:
                    visible = True
            elif laps2[i,2] == -1:
                label = label + " W"
                if init_state and HIDE_ALL_NOT_BEST_OR_WORST and display_mode != DISPLAY_MODE_PEDALS:
                    visible = True
            
            if display_mode == DISPLAY_MODE_PEDALS:
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(pedal_pos.min(), pedal_pos.max())
                lc = LineCollection(segments, cmap=pedal_cmap, norm=norm)
                lc.set_array(pedal_pos)
                lc.set_linewidth(line_width)
                lc.set_alpha(line_alpha)
                lc.set_label(label)
                lc.set_visible(visible)
                lc.set_transform(tr)
                ln = ax.add_collection(lc)
            elif display_mode == DISPLAY_MODE_VELOCITY:
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(np.amin(velocities), np.amax(velocities))
                lc = LineCollection(segments, cmap=velocity_cmap, norm=norm)
                lc.set_array(velocities)
                lc.set_linewidth(line_width)
                lc.set_alpha(line_alpha)
                lc.set_label(label)
                lc.set_visible(visible)
                lc.set_transform(tr)
                ln = ax.add_collection(lc)
            elif display_mode == DISPLAY_MODE_CURVATURE:
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                cmin = np.amin(curvatures)
                cmax = np.amax(curvatures)
                absmax = max(np.abs(cmin), np.abs(cmax))
                norm = plt.Normalize(-absmax, absmax)
                lc = LineCollection(segments, cmap=curvature_cmap, norm=norm)
                lc.set_array(curvatures)
                lc.set_linewidth(line_width)
                lc.set_alpha(line_alpha)
                lc.set_label(label)
                lc.set_visible(visible)
                lc.set_transform(tr)
                ln = ax.add_collection(lc)
                
            elif display_mode == DISPLAY_MODE_VELOCITY_PLOT:
                x2 = 100*traj[:-2,3]  # Track Position in %
                y2 = 3.6*traj[:-2,7]  # Velocity in km/h
                veloc_plot.add_series(x2, y2, None, line_color, visible, label)
                current_plot = veloc_plot
                '''
                if laps2[i,2] == 1:
                    tr2 = mtransforms.Affine2D().rotate_deg(CIRCUIT_ROTATION).translate(CIRCUIT_TRANSLATE_X, CIRCUIT_TRANSLATE_Y) + ax_inset.transData
                    veloc_plot.add_inset(ax_inset, tr2)
                    veloc_plot.add_subseries(x, y, (0.5,0.5,0.5))
                
                ln = None
                '''    
                
                
            elif display_mode == DISPLAY_MODE_G_PLOT:
                tp = 100*traj[:-2,3]  # Track Position in %
                acc_lat = traj[:-2,8]      # Accel Lateral in g
                acc_long = traj[:-2,9]      # Accel Longitudial in g
                theta = np.arctan2(acc_long, acc_lat)
                rad = np.sqrt(acc_lat**2 + acc_long**2)
                g_plot.add_series(tp, theta, rad, line_color, visible, label)
                current_plot = g_plot
                '''
                if laps2[i,2] == 1:
                    tr2 = mtransforms.Affine2D().rotate_deg(CIRCUIT_ROTATION).translate(CIRCUIT_TRANSLATE_X, CIRCUIT_TRANSLATE_Y) + ax_inset.transData
                    g_plot.add_inset(ax_inset, tr2)
                    g_plot.add_subseries(x, y, (0.5,0.5,0.5))
                 
                ln = None
                '''
            
            elif display_mode == DISPLAY_MODE_TRAJ_DIST:
                tp_traj = valid_tp_trajectories[i]
                x2 = 100*tp_traj[:,3]  # Track Position in %
                px = tp_traj[:,0]   # World pos X
                pz = -tp_traj[:,2]  # World pos Z
                dx = px - blpx
                dz = pz - blpz
                
                proj_disp = dx*blvdirx + dz*blvdirz
                lateral_proj_disp = -dx*blvdirz + dz*blvdirx
                
                #vx = tp_traj[:,10]  # Best lap velocity X in m/s
                #vz = -tp_traj[:,12]  # Best lap velocity Z in m/s
                #v_ang = np.arctan2(vz, vx) * 180 / np.pi
                #dist = np.sqrt(dx*dx + dz*dz) # Distance to best lap in meters
                
                traj_disp_plot.add_series(x2, lateral_proj_disp, None, line_color, visible, label)
                current_plot = traj_disp_plot
                '''
                if laps2[i,2] == 1:
                    x3 = tp_traj[:,0]
                    y3 = -tp_traj[:,2]
                    tr2 = mtransforms.Affine2D().rotate_deg(CIRCUIT_ROTATION).translate(CIRCUIT_TRANSLATE_X, CIRCUIT_TRANSLATE_Y) + ax_inset.transData
                    traj_disp_plot.add_inset(ax_inset, tr2)
                    traj_disp_plot.add_subseries(x3, y3, (0.5,0.5,0.5))
                    
                ln = None
                '''
                
            elif display_mode == DISPLAY_MODE_VELOC_ANGLE:
                tp_traj = valid_tp_trajectories[i]
                x2 = 100*tp_traj[:,3]  # Track Position in %
                
                # TODO add plot with car angle |||| this is trajectory direction angle
                vx = tp_traj[:,10]  # Best lap velocity X in m/s
                vz = -tp_traj[:,12]  # Best lap velocity Z in m/s
                v_ang = np.arctan2(vz, vx) * 180 / np.pi
                
                delta_ang = v_ang - blv_ang
                delta_ang = np.where(delta_ang < 180, delta_ang + 360, delta_ang)
                delta_ang = np.where(delta_ang > 180, delta_ang - 360, delta_ang)
                
                veloc_angle_plot.add_series(x2, delta_ang, None, line_color, visible, label)
                current_plot = veloc_angle_plot
                
            elif display_mode == DISPLAY_MODE_TIRE_SLIP_ANGLE:
                tp_traj = valid_tp_trajectories[i]
                x2 = 100*tp_traj[:,3]  # Track Position in %
                
                # 13:SlipAngleW  14:SlipAngleX  15:SlipAngleY  16:SlipAngleZ
                for j in range(0,4):
                    slip_ang = tp_traj[:,13+j]
                    slip_angle_plot.add_series(x2, slip_ang, None, line_color, visible, label)
                
                current_plot = slip_angle_plot
                
            elif display_mode == DISPLAY_MODE_TIRE_SLIP_RATIO:
                tp_traj = valid_tp_trajectories[i]
                x2 = 100*tp_traj[:,3]  # Track Position in %
                
                # 17:SlipRatioW  18:SlipRatioX  19:SlipRatioY  20:SlipRatioZ
                for j in range(0,4):
                    slip_ratio = tp_traj[:,17+j]
                slip_ratio_plot.add_series(x2, slip_ratio, None, line_color, visible, label)
                current_plot = slip_ratio_plot
             
            elif display_mode == DISPLAY_MODE_LIDAR:
                ax.plot(sideleft_xy[:,0],  sideleft_xy[:,1],  color='k', linewidth=0.5, alpha=0.5, visible=visible, label="side_left", transform=tr)
                ax.plot(sideright_xy[:,0], sideright_xy[:,1], color='k', linewidth=0.5, alpha=0.5, visible=visible, label="side_right", transform=tr)
                #ln, = ax.plot(x, y, color=line_color, linewidth=line_width, alpha=line_alpha, visible=visible, label=label, transform=tr)
                
                #tp_traj = valid_tp_trajectories[i]
                #x2 = 100*tp_traj[:,3]  # Track Position in %
                
                # 13:SlipAngleW  14:SlipAngleX  15:SlipAngleY  16:SlipAngleZ
                #for j in range(0,4):
                #    slip_ang = tp_traj[:,13+j]
                #    lidar_plot.add_series(x2, slip_ang, None, line_color, visible, label)
                
                #y2 = 3.6*tp_traj[:,7]  # Velocity in km/h
                #lidar_plot.add_series(x2, y2, None, line_color, visible, label)
                #current_plot = lidar_plot
                
                
                #x2 = 100*traj[:-2,3]  # Track Position in %
                #y2 = 3.6*traj[:-2,7]  # Velocity in km/h
                lidar_plot.add_series(x, y, None, line_color, visible, label)
                current_plot = lidar_plot
                
               
            else:
                ln, = ax.plot(x, y, color=line_color, linewidth=line_width, alpha=line_alpha, visible=visible, label=label, transform=tr)
                
                
            if display_mode in [DISPLAY_MODE_VELOCITY_PLOT, DISPLAY_MODE_G_PLOT, DISPLAY_MODE_TRAJ_DIST, DISPLAY_MODE_VELOC_ANGLE, DISPLAY_MODE_TIRE_SLIP_ANGLE, DISPLAY_MODE_TIRE_SLIP_RATIO, DISPLAY_MODE_LIDAR]:
                if laps2[i,2] == 1:
                    tr2 = mtransforms.Affine2D().rotate_deg(CIRCUIT_ROTATION).translate(CIRCUIT_TRANSLATE_X, CIRCUIT_TRANSLATE_Y) + ax_inset.transData
                    current_plot.add_inset(ax_inset, tr2)
                    
                    if display_mode in [DISPLAY_MODE_VELOCITY_PLOT, DISPLAY_MODE_G_PLOT, DISPLAY_MODE_LIDAR]:
                        current_plot.add_subseries(x, y, (0.5,0.5,0.5))
                    elif display_mode in [DISPLAY_MODE_TRAJ_DIST, DISPLAY_MODE_VELOC_ANGLE, DISPLAY_MODE_TIRE_SLIP_ANGLE, DISPLAY_MODE_TIRE_SLIP_RATIO]:
                        x3 = tp_traj[:,0]
                        y3 = -tp_traj[:,2]
                        current_plot.add_subseries(x3, y3, (0.5,0.5,0.5))

                #if display_mode != DISPLAY_MODE_LIDAR:
                ln = None
                
                
            if ln is not None:
                laplines.append(ln)
            
    minv = np.amin(velocities)
    maxv = np.amax(velocities)
    if minv < min_veloc:
        min_veloc = minv
    if maxv > max_veloc:
        max_veloc = maxv
        
    minc = np.amin(curvatures)
    maxc = np.amax(curvatures)
    if minc < min_curv:
        min_curv = minc
    if maxc > max_curv:
        max_curv = maxc
    
    '''
    if display_mode == DISPLAY_MODE_VELOCITY_PLOT:    
        plotter.plot(veloc_plot)
        for i in range(len(plotter.ln)):
            ln = plotter.ln[i][0]
            laplines.append(ln)
            
    elif display_mode == DISPLAY_MODE_G_PLOT:    
        plotter.plot(g_plot)
        for i in range(len(plotter.ln)):
            ln = plotter.ln[i][0]
            laplines.append(ln)
    '''      
    if display_mode in [DISPLAY_MODE_VELOCITY_PLOT, DISPLAY_MODE_G_PLOT, DISPLAY_MODE_TRAJ_DIST, DISPLAY_MODE_VELOC_ANGLE, DISPLAY_MODE_TIRE_SLIP_ANGLE, DISPLAY_MODE_TIRE_SLIP_RATIO, DISPLAY_MODE_LIDAR]:
        plotter.plot(current_plot)
        for i in range(len(plotter.ln)):
            ln = plotter.ln[i][0]
            laplines.append(ln)
            
    else:
        plotter.clear()

    ax.text(0.5, 0.93, FOLDER + " - " + " - ".join(FILE_FILTER), fontsize=12, ha="center", transform=ax.transAxes)

    if display_mode == DISPLAY_MODE_VELOCITY_PLOT:
        ax.set_xlabel('Lap Progression [%]')
        ax.set_ylabel('Velocity [km/h]')
        
    if display_mode == DISPLAY_MODE_TRAJ_DIST:
        ax.set_xlabel('Lap Progression [%]')
        ax.set_ylabel('Lateral Distance [m]')
        
    if display_mode == DISPLAY_MODE_VELOC_ANGLE:
        ax.set_xlabel('Lap Progression [%]')
        ax.set_ylabel('Velocity Angle Delta [°]')
        
    if display_mode == DISPLAY_MODE_TIRE_SLIP_ANGLE:
        ax.set_xlabel('Lap Progression [%]')
        ax.set_ylabel('Tire Slip Angle [°]')
        
    if display_mode == DISPLAY_MODE_TIRE_SLIP_RATIO:
        ax.set_xlabel('Lap Progression [%]')
        ax.set_ylabel('Tire Slip Ratio [%]')
        
    if display_mode == DISPLAY_MODE_LIDAR:
        ax.set_xlabel('TODO')
        ax.set_ylabel('TODO')
        
        
    # make colorbar scale
    if DISPLAY_ONLY_FULL_LAPS and display_mode in [DISPLAY_MODE_LAPTIME, DISPLAY_MODE_PEDALS, DISPLAY_MODE_VELOCITY, DISPLAY_MODE_CURVATURE]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1.5%", pad=0)
        
        if display_mode == DISPLAY_MODE_LAPTIME:
            sm = cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(vmin=0, vmax=0.001*lap_delta))
            cb = plt.colorbar(sm, cax=cax, format='%.1f sec')
            cb.ax.tick_params(labelsize=7)        
        elif display_mode == DISPLAY_MODE_PEDALS:
            sm = cm.ScalarMappable(cmap=pedal_cmap, norm=plt.Normalize(vmin=0, vmax=1))
            plt.colorbar(sm, cax=cax, format='%.1f')
        elif display_mode == DISPLAY_MODE_VELOCITY:
            sm = cm.ScalarMappable(cmap=velocity_cmap, norm=plt.Normalize(vmin=min_veloc*3.6, vmax=max_veloc*3.6))
            cb = plt.colorbar(sm, cax=cax, format='%.1f km/h')
            cb.ax.tick_params(labelsize=7, pad=-70, left=True, right=False, direction="out")
        elif display_mode == DISPLAY_MODE_CURVATURE:
            sm = cm.ScalarMappable(cmap=curvature_cmap, norm=plt.Normalize(vmin=min_curv, vmax=max_curv))
            cb = plt.colorbar(sm, cax=cax, format='%.1f')


def config_checkbuttons(fig):
    global laplines
    
    # make checkbuttons
    ax = fig.axes[0]
    #ax.set_aspect('equal')
    #ax.set_frameon(False)
    ax.axis('off')
    np_lap = len(laplines)

    line_labels = [str(line.get_label()) for line in laplines]
    visibility = [line.get_visible() for line in laplines]
    check = CheckButtons(ax, line_labels, visibility)
    #check.eventson = False

    figW, figH = ax.get_figure().get_size_inches()
    _, _, aw, ah = ax.get_position().bounds
    ax_ar = (figH * ah) / (figW * aw)

    #print("Nb laps: %d" % (np_lap))

    for i in range(np_lap):
        xy = list(check.rectangles[i].get_xy())
        size = 0.3*np.log10(np_lap) / np_lap
        check.rectangles[i].set_width(size * ax_ar)
        check.rectangles[i].set_height(size)
        check.labels[i].set_horizontalalignment("left")
        py = xy[1]
        py = py + 0.5*size
        pos = list(check.labels[i].get_position())
        pos[0] = 0.15 + size * ax_ar
        pos[1] = py
        check.labels[i].set_position(tuple(pos))
        check.labels[i].set_va("center")

        for ll in check.lines[i]:
            ll.set_linewidth(2.0)
            ll.set_color((0.2, 0.2, 0.2))
            ll.set_alpha(0.7)
            xdata = ll.get_xdata()
            xdata[1] = xdata[0] + size * ax_ar
            ll.set_xdata(xdata)
            ydata = ll.get_ydata()
            if (ydata[0] > ydata[1]):
                ydata[0] = ydata[1] + size
            else:
                ydata[1] = ydata[0] + size
            ll.set_ydata(ydata)
            
            ll.set_visible(visibility[i])

        if laps2[i,2] == 1:
            check.rectangles[i].set_facecolor("green")
            check.rectangles[i].set_edgecolor("k")
            check.rectangles[i].set_alpha(0.2)
            check.labels[i].set_color("green")
        elif laps2[i,2] == -1:
            check.rectangles[i].set_facecolor("red")
            check.rectangles[i].set_edgecolor("k")
            check.rectangles[i].set_alpha(0.2)
            check.labels[i].set_color("red")

    return check


def ckbtn_onclick(label):
    global laplines, checkbtn, plotter
    
    line_labels = [str(line.get_label()) for line in laplines]
    index = line_labels.index(label)
    status = checkbtn.get_status()
    laplines[index].set_visible(status[index])
    
    if plotter is not None:
        pln_size = len(plotter.ln)
        if index < pln_size:
            
            if (plotter.obj is not None) and (plotter.obj.mode == 'classic'):
                plotter.ln[index][0].set_visible(status[index])
                plotter.dot[index][0].set_visible(status[index])
            vis_plot = plotter.get_plot()
            if vis_plot is not None:
                vis_plot.visible[index] = status[index]
            #print(index, status[index], vis_plot.visible)
    
        if (plotter.obj is not None) and (plotter.obj.mode == 'polar'):
            plotter.update_polar_plot(plotter.obj.trackpos)
    
    #print(status[index], laplines[index].get_visible(), checkbtn.lines[index][0].get_visible())
    plt.draw()
    
    
    

#=============================================================================


#==================
# Main plot code  #
#==================

fig = plt.figure(figsize=(16,8))
gs = fig.add_gridspec(8, 8)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 3:6])
ax3 = fig.add_subplot(gs[1:, 1:], label=DISPLAY_MODE)

# make plot
config_plot(ax3, DISPLAY_MODE)

# make checkbuttons
checkbtn = config_checkbuttons(fig)

# make radiobuttons
ax2.axis('off')
radio =  MyRadioButtons(fig, ax2,['Lap Time','Pedals','Velocity Track', 'Curvature', 'Velocity Plot', 'G Plot', 'Traj Lateral Dist', 'Veloc Delta Angle', 'Tire Slip Angle', 'Tire Slip Ratio', 'Lidar'], active=1, activecolor='green', size=100, ncol=3)

radio.set_checkbuttons(checkbtn)
checkbtn.on_clicked(ckbtn_onclick)

plt.tight_layout()
plt.show()




