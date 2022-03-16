import math 
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import os
import pandas as pd
import matplotlib.transforms as mtransforms
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.widgets import AxesWidget, CheckButtons, RadioButtons, Slider
from scipy.interpolate import interp1d
import time 
# lidar 2D
# interesctions based on https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
global laplines, checkbtn, plotter, LINE_WIDTH, LINE_ALPHA, LINE_COLOR
def init_track_data(track = 'vallelunga',folder="Data",plot=False):
    TRACK = track

    FOLDER = folder


    FOLDER = FOLDER + "/" + TRACK



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



    CIRCUIT_LIMIT_X = 560       # in meters     560 for Vallelunga
    CIRCUIT_LIMIT_Y = 195       # in meters     195 for Vallelunga
    CIRCUIT_ROTATION = 0     # in degrees   -57° for Vallelunga
    CIRCUIT_TRANSLATE_X = 0     # in meters       0 for Vallelunga
    CIRCUIT_TRANSLATE_Y = 0    # in meters       0 for Vallelunga

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

    # arrays of 2D projected world coordinate frame vertors   (x2d = x3d   y2d = - z3d)
    centerline_xy = []
    sideleft_xy = []
    sideright_xy = []



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
    

    
        # project 3D coordinates arrays into 2D coordinates
        centerline_xy = np.delete(centerline_xyz, obj=1, axis=1)
        centerline_xy[:,1] *= -1
        sideleft_xy = np.delete(sideleft_xyz, obj=1, axis=1)
        sideleft_xy[:,1] *= -1
        sideright_xy = np.delete(sideright_xyz, obj=1, axis=1)
        sideright_xy[:,1] *= -1
    if plot:
        df= pd.read_excel('test_PID.xlsx')
        df = df.sort_values(by='spline_position')
        positions =[]
        for i in range(df.shape[0]):
            temp=df.position[i].replace(' ','').replace('[','').replace(']','').split(',')
            temp_list=[]
            for idx,j in enumerate(temp):
                value=float(j)
                if idx== 2:
                    value*=-1
            
                temp_list.append(value)
            positions.append(temp_list)
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot()

        tr = mtransforms.Affine2D().rotate_deg(CIRCUIT_ROTATION).translate(CIRCUIT_TRANSLATE_X, CIRCUIT_TRANSLATE_Y) + ax.transData
        
        scatter =ax.scatter(list(zip(*positions))[0], list(zip(*positions))[2], c=df.speedKmh, marker="o", s=0.2*(72./fig.dpi)**2, transform=tr)
        ax.annotate(str(df.lap_time_ms[int(df.shape[0])-200]/1000)+' Sec', (list(zip(*positions))[0][int(df.shape[0]-200)], list(zip(*positions))[2][int(df.shape[0]-200)]))
        ax.annotate(str(df.lap_time_ms[int(df.shape[0]*2/6)]/1000)+' Sec', (list(zip(*positions))[0][int(df.shape[0]*2/6)], list(zip(*positions))[2][int(df.shape[0]*2/6)]))
        ax.annotate(str(df.lap_time_ms[int(df.shape[0]*3/6)]/1000)+' Sec', (list(zip(*positions))[0][int(df.shape[0]*3/6)], list(zip(*positions))[2][int(df.shape[0]*3/6)]),xytext=(list(zip(*positions))[0][int(df.shape[0]*3/6)]+10, list(zip(*positions))[2][int(df.shape[0]*3/6)]))
        ax.annotate(str(df.lap_time_ms[int(df.shape[0]*4/6)]/1000)+' Sec', (list(zip(*positions))[0][int(df.shape[0]*4/6)], list(zip(*positions))[2][int(df.shape[0]*4/6)]))
        ax.annotate(str(df.lap_time_ms[int(df.shape[0]*5/6)]/1000)+' Sec', (list(zip(*positions))[0][int(df.shape[0]*5/6)], list(zip(*positions))[2][int(df.shape[0]*5/6)]))
        #ax.annotate('Start Point', (list(zip(*positions))[0][0], list(zip(*positions))[2][0]))
        #ax.annotate('End Point', (list(zip(*positions))[0][-1], list(zip(*positions))[2][-1]))
        ax.scatter(centerline_xy[:,0], centerline_xy[:,1], c="#000000", marker="o", s=0.2*(72./fig.dpi)**2, transform=tr,label='Centerline')
        ax.scatter(sideleft_xy[:,0], sideleft_xy[:,1], c="r", marker="o", s=0.2*(72./fig.dpi)**2, transform=tr,label='left side')
        ax.scatter(sideright_xy[:,0], sideright_xy[:,1], c="b", marker="o", s=0.2*(72./fig.dpi)**2, transform=tr,label='right side')
        legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower right", title="Speed")
        ax.add_artist(legend1)
        ax.legend()
        ax.set_title('Progression of the Agent on Vallelunga')
        plt.show()

    return sideleft_xy,sideright_xy,centerline_xy,forward_xyz,track_data
    
#init_track_data(plot=True)    


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
def compute_lidar(max_dist, half_angle, angle_inc, dir_xy,intersect_type, size,p1_array, p2_array, q2_array, p3_array, q3_array):
    
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
# compile the list of valid laps

#interp_kind = 'linear' #'quadratic'

#traj = pd.read_excel('test.xlsx')

        
## apply linear interpolation to align data with track position (tp) instead of time
#tp = traj['spline_position']   # trackpos in %
        
## correct last 5 tp data if below 50%  (correction for tp rolled back to zero)
#tp_tmp = tp[-5:]
#tp_tmp = np.where(tp_tmp < 0.5, tp_tmp + 1.0, tp_tmp)
#tp[-5:] = tp_tmp
#tp_tick = 0.0005   # 0.05%   =>  2000 data point per lap  ~=  every 2.3 m in Barcelona
#tp2 = np.arange(0.0, 1.0 + tp_tick, tp_tick)
        
#tp_traj = []
#for i in traj.columns:
#    if isinstance(traj['{}'.format(i)][0], str):
#        if ',' in traj['{}'.format(i)][0]:
#            traj['{}'.format(i)] = traj['{}'.format(i)].apply(lambda r: np.fromstring(r[1:-1],sep=','))
#        else:
#            traj['{}'.format(i)] = traj['{}'.format(i)].apply(lambda r: np.fromstring(r[1:-1],sep=' '))

#traj["WorldPosition_X"] = traj.position.apply(lambda x: x[0])
#traj["WorldPosition_Y"] = traj.position.apply(lambda x: x[1])
#traj["WorldPosition_Z"] = traj.position.apply(lambda x: x[2])
#traj["LocalVelocity_X"] = traj.local_velocity.apply(lambda x: x[0])
#traj["LocalVelocity_Y"] = traj.local_velocity.apply(lambda x: x[1])
#traj["LocalVelocity_Z"] = traj.local_velocity.apply(lambda x: x[2])
#traj['AccG_X'] = traj.acceleration.apply(lambda x: x[0])
#traj['AccG_Z'] = traj.acceleration.apply(lambda x: x[2])
#traj["Velocity_X"] = traj.velocity.apply(lambda x: x[0])
#traj["Velocity_Y"] = traj.velocity.apply(lambda x: x[1])
#traj["Velocity_Z"] = traj.velocity.apply(lambda x: x[2])
#traj['SlipAngle_W'] = traj.slip_angle_deg.apply(lambda x: x[0])
#traj['SlipAngle_X'] = traj.slip_angle_deg.apply(lambda x: x[1])
#traj['SlipAngle_Y'] = traj.slip_angle_deg.apply(lambda x: x[2])
#traj['SlipAngle_Z'] = traj.slip_angle_deg.apply(lambda x: x[3])
#traj['SlipRatio_W'] = traj.slip.apply(lambda x: x[0])
#traj['SlipRatio_X'] = traj.slip.apply(lambda x: x[1])
#traj['SlipRatio_Y'] = traj.slip.apply(lambda x: x[2])
#traj['SlipRatio_Z'] = traj.slip.apply(lambda x: x[3])
#columns_to_keep = ['WorldPosition_X',"WorldPosition_Y","WorldPosition_Z",'spline_position','lap_time_ms','gas','brake',"LocalVelocity_X","LocalVelocity_Y","LocalVelocity_Z",
#                   'AccG_X','AccG_Z',"Velocity_X","Velocity_Y","Velocity_Z",'SlipAngle_W','SlipAngle_X','SlipAngle_Y','SlipAngle_Z','SlipRatio_W','SlipRatio_X','SlipRatio_Y','SlipRatio_Z']
#traj = traj[columns_to_keep]
#for i in traj.columns:
#    y = traj['{}'.format(i)]
#    f = interp1d(tp, y, kind=interp_kind, bounds_error=False, fill_value="extrapolate")
#    y2 = f(tp2)
#    tp_traj.append(y2)
        
#tp_traj = np.array(tp_traj).transpose()
#valid_tp_trajectories.append(tp_traj)


#print(lidar_segments.shape)
def compute_distances(position,lidar_points,centralize):
    lidar_distances = []
    for i in lidar_points:
        i[1] *= -1
        distance=np.linalg.norm(i-[position[0],position[2]])
        #if distance > 300:
        #    distance = 300
        if centralize:
            lidar_distances.append(np.log(distance)) 
        else:
            lidar_distances.append(distance)
    arr = np.array(lidar_distances)
    arr = arr.astype(np.float32)
    return arr

def compute_lidar_distances(look,position,sideleft_xy,sideright_xy,centralize=False):
    max_dist = 3000  # 1km forward looking
    half_angle = 60
    angle_inc = 2


    size = sideleft_xy.shape[0]

    intersect_type = np.single

    # vectorize intersection computation
    centerline_xy = [position[0],-position[2]]
    p1_array = [centerline_xy[0], centerline_xy[1]]
    p1_array = np.array(p1_array)[np.newaxis].astype(intersect_type)
    p1_array = np.repeat(p1_array, repeats=size, axis=0)

    p2_array = sideleft_xy.astype(intersect_type)
    q2_array = np.roll(p2_array,-1, axis=0)

    p3_array = sideright_xy.astype(intersect_type)
    q3_array = np.roll(p3_array,-1, axis=0)

    #becomes look vector
    dir_xy = [look[0], -look[2]]

    lidar_segments = compute_lidar(max_dist, half_angle, angle_inc, dir_xy,intersect_type,size ,p1_array, p2_array, q2_array, p3_array, q3_array)
    lidar_points = []
    for i in lidar_segments:
        lidar_points.append(i[1])
    lidar_points= np.array(lidar_points)
    arr = compute_distances(position,lidar_points,centralize)
    ####################   PLOT
    #CIRCUIT_LIMIT_X = 560       # in meters     560 for Vallelunga
    #CIRCUIT_LIMIT_Y = 195       # in meters     195 for Vallelunga
    #CIRCUIT_ROTATION = 0     # in degrees   -57° for Vallelunga
    #CIRCUIT_TRANSLATE_X = 0     # in meters       0 for Vallelunga
    #CIRCUIT_TRANSLATE_Y = 0    # in meters       0 for Vallelunga
    #fig = plt.figure(figsize=(16,8))
    #ax = fig.add_subplot()
    #tr = mtransforms.Affine2D().rotate_deg(CIRCUIT_ROTATION).translate(CIRCUIT_TRANSLATE_X, CIRCUIT_TRANSLATE_Y) + ax.transData
    #ax.scatter(sideleft_xy[:,0], sideleft_xy[:,1], c="r", marker="o", s=0.2*(72./fig.dpi)**2, transform=tr,label='left side')
    #ax.scatter(sideright_xy[:,0], sideright_xy[:,1], c="b", marker="o", s=0.2*(72./fig.dpi)**2, transform=tr,label='right side')
    #ax.set_title('Lidar sensors')
    #if lidar_segments.shape[0] > 0:
    #    for k in range(lidar_segments.shape[0]):
    #        ax.plot(lidar_segments[k,:,0], lidar_segments[k,:,1], c="m", linewidth=0.5, alpha=0.5, transform=tr)
    return arr,lidar_points,centerline_xy
