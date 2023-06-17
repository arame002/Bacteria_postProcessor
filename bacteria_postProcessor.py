import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.lines as mlines
import matplotlib.colors as colors
import math
import pandas as pd
import trackpy as tp
from functools import reduce
from matplotlib.ticker import PercentFormatter
import matplotlib.gridspec as gridspec
import random
from scipy import stats
import multiprocessing


#define functions

# We may want to use third-order Savitzky-Golay filter to detect reversals.
# It is based on local curvature of the trajectory

# Develope Kolmogorov–Smirnov method to see if we have 2 distributions for turning angles

def calculate_velocity_forward(x, y, dt,gap):
    velocity = np.zeros( math.ceil(len(x) / gap ) - 1 )
    for i in range(0, len(x) - gap, gap):
        velocity[i // gap] = np.sqrt((x[i+gap] - x[i])**2 + (y[i+gap] - y[i])**2) / (dt * gap)
    return velocity

def calculate_velocity_backward(x, y, dt, gap):
    velocity = np.zeros( math.ceil(len(x) / gap ) - 1 )
    for i in range(gap, len(x), gap):
        velocity[i// gap - 1] = np.sqrt((x[i] - x[i-gap])**2 + (y[i] - y[i-gap])**2) / (dt * gap)
    return velocity

def calculate_velocity_central(x, y, dt, gap):
    velocity = np.zeros(math.ceil(len(x) / gap) -2 )
    for i in range(gap, len(x) - gap, gap):
        velocity[(i-gap)//gap] = np.sqrt((x[i+gap] - x[i-gap])**2 + (y[i+gap] - y[i-gap])**2) / (2 * gap * dt)
    return velocity
    
def calculate_angular_velocity_forward(x, y, dt, gap):
    vx = np.zeros( math.ceil(len(x) / gap ) - 1 )
    vy = np.zeros( math.ceil(len(x) / gap ) - 1 )
    for i in range(0, len(x) - gap, gap):
        vx[i // gap] = (x[i+gap] - x[i]) / (dt * gap)
        vy[i // gap] = (y[i+gap] - y[i]) / (dt * gap)
    theta = np.arctan2(vy, vx)
    angular_velocity = (theta[1:] - theta[:-1]) / (dt * gap)
    return np.abs(angular_velocity)

def calculate_angular_velocity_backward(x, y, dt, gap):
    vx = np.zeros( math.ceil(len(x) / gap ) - 1 )
    vy = np.zeros( math.ceil(len(x) / gap ) - 1 )
    for i in range(gap, len(x), gap):
        vx[i // gap - 1] = (x[i] - x[i-gap]) / (dt * gap)
        vy[i // gap - 1] = (y[i] - y[i-gap]) / (dt * gap)
    theta = np.arctan2(vy, vx)
    angular_velocity = (theta[1:] - theta[:-1]) / (dt * gap)
    return np.abs(angular_velocity)

def calculate_angular_velocity_central(x, y, dt, gap):
    vx = np.zeros(math.ceil(len(x) / gap) -2 )
    vy = np.zeros(math.ceil(len(x) / gap) -2 )
    for i in range(gap, len(x) - gap, gap):
        #vx[(i-gap)//gap] = (x[i+gap] - x[i-gap]) / (2 * gap * dt)
        #vy[(i-gap)//gap] = (y[i+gap] - y[i-gap]) / (2 * gap * dt)
        vx[(i-gap)//gap] = (x[i+gap] - x[i]) / (1.0 * gap * dt)
        vy[(i-gap)//gap] = (y[i+gap] - y[i]) / (1.0 * gap * dt)
    theta = np.arctan2(vy, vx)
    angular_velocity = (theta[1:] - theta[:-1])
    angular_velocity = np.abs(angular_velocity)
    #angular_velocity = (angular_velocity[:-1] + angular_velocity[1:] )/2.0
    angular_velocity = np.where(angular_velocity > math.pi , 2.0 * math.pi - angular_velocity, angular_velocity)
    #angular_velocity = angular_velocity / (2 * gap * dt)
    return np.abs(angular_velocity / (1.0 * gap * dt) )
    
def calculate_angular_velocity_central2(x, y, dt, gap):
    vx = np.zeros(math.ceil(len(x) / gap) -2 )
    vy = np.zeros(math.ceil(len(x) / gap) -2 )
    for i in range(gap, len(x) - gap, gap):
        #vx[(i-gap)//gap] = (x[i+gap] - x[i-gap]) / (2 * gap * dt)
        #vy[(i-gap)//gap] = (y[i+gap] - y[i-gap]) / (2 * gap * dt)
        vx[(i-gap)//gap] = (x[i+gap] - x[i]) / (1.0 * gap * dt)
        vy[(i-gap)//gap] = (y[i+gap] - y[i]) / (1.0 * gap * dt)
    cosijk = np.zeros_like(vx[:-1])
    for i in range(len(cosijk) ):
        cosijk[i] = vx[i]*vx[i+1] + vy[i]*vy[i+1]
        cosijk[i]  /= ( math.sqrt(vx[i]**2 + vy[i]**2)  + 1e-10)
        cosijk[i]  /= ( math.sqrt(vx[i+1]**2 + vy[i+1]**2) + 1e-10 )
    
    theta = np.arccos(cosijk)
    angular_velocity = theta / (1.0 * gap * dt)
    #angular_velocity = np.where(angular_velocity > math.pi , 2.0 * math.pi - angular_velocity, angular_velocity)
    #angular_velocity = angular_velocity / (2 * gap * dt)
    return np.abs(angular_velocity )

def find_events(angular_velocity, threshold):
    events = np.where(angular_velocity >= threshold)[0]
    return events
    
def remove_data(x, y, times,velocity, angular_velocity, events, gap,remove_size=0):
    if events.size < 2:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    start = events[0]
    end = events[-1]
    #Modify this to get rid of short events
    angular_velocity = angular_velocity[start * gap + remove_size :end * gap +1 - remove_size]
    velocity = velocity[start * gap + gap + remove_size:end * gap +1 + gap - remove_size]
    times = times[start * gap + 2 * gap + remove_size:end * gap +1 + 2 * gap - remove_size]
    x = x[start * gap + 2 * gap + remove_size:end * gap +1 + 2 * gap - remove_size]
    y = y[start * gap + 2 * gap + remove_size:end * gap +1 + 2 * gap - remove_size]
    return x,y,times,velocity,angular_velocity
    
def find_max_pairwise_distance(x, y, threshold):
    x = np.ravel(x)
    y = np.ravel(y)
    pairwise_distance = np.abs((x - x[:, None])**2 + (y - y[:, None])**2)
    max_distance = np.amax(pairwise_distance)
    return max_distance > threshold

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def smooth_rectangular(x, window_size):
    """Smooth a 1D signal using the rectangular method."""
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(x, window, mode='same')
    return smoothed
    
def plot_with_multicolor(xdata, ydata, colors, axs,id,show_text=True, linestyle='-'):
    for i in range(len(xdata) - 1):
        axs[id].plot(xdata[i:i+2], ydata[i:i+2], color=colors[i],linestyle=linestyle)
        if show_text :
            for i in range(len(xdata)-1):
                axs[id].text(xdata[i], ydata[i], str(i), ha='center', va='center', fontsize=3)
                
def merge_trajectories(prt1, prt2,id1,id2,distance_threshold,frame_threshold):
    # Calculate the distance between the final position of df1 and the initial position of df2
    #distance = np.sqrt((prt1[-1][8] - prt2[0][8])**2 + (prt1[-1][9] - prt2[0][9])**2)
    distance = np.sqrt((prt1[-1][3] - prt2[0][3])**2 + (prt1[-1][4] - prt2[0][4])**2)
    # Calculate the time gap between the end of df1 and the start of df2
    time_gap = prt2[0][5] - prt1[-1][5]
    # Merge the trajectories if they meet the criteria
    if distance <= distance_threshold and time_gap <= frame_threshold:
        return (id1,id2)
    else:
        return None

def max_consecutive_below_threshold(arr, threshold):

    consecutive_counts = 0
    max_consecutive_counts = 0

    for i in range(len(arr)):
        if arr[i] < threshold:
            consecutive_counts += 1
            if consecutive_counts > max_consecutive_counts:
                max_consecutive_counts = consecutive_counts
        else:
            consecutive_counts = 0

    return max_consecutive_counts

def plot_Trajectory_Panel(particle_data, particle,dir_name, gap1 ) :

        times = [d[0] for d in particle_data]
        velocity_central1 = [d[1] for d in particle_data]
        angular_velocity_central1 = [d[2] for d in particle_data]
        x = [d[3] for d in particle_data]
        y = [d[4] for d in particle_data]
        frames = [d[5] for d in particle_data]
        velocity_central2 = [d[6] for d in particle_data]
        angular_velocity_central2 = [d[7] for d in particle_data]
        x_moving_average = [d[8] for d in particle_data]
        y_moving_average = [d[9] for d in particle_data]
        
        
        fig = plt.figure(figsize=(10,6))
        gs = fig.add_gridspec(2, 2)
        axs = []
        #axs.append(fig.add_subplot(gs[:, 0]))
        axs.append(fig.add_subplot(gs[0, 0]))
        axs.append(fig.add_subplot(gs[0, 1]))
        axs.append(fig.add_subplot(gs[1, 0]))
        axs.append(fig.add_subplot(gs[1, 1]))
        # Create the color map
        cmap = plt.get_cmap('viridis')
        n_colors = cmap.N
        colors1 = cmap(np.linspace(0, 1, len(times) ))
        norm = colors.Normalize(vmin=min(velocity_central2), vmax=max(velocity_central2))
        normalized_v = norm(velocity_central2)
        colors2 = cmap(normalized_v)
        colorss = colors2
        for i in range(len(x) - 1):
            axs[0].plot(x[i:i+2], y[i:i+2], color=colorss[i], linestyle="-")
            #axs[0].plot(x_moving_average[i:i+2], y_moving_average[i:i+2], color=colorss[i], linestyle="-",alpha=0.5)
        #plot_with_multicolor(x_moving_average,y_moving_average,colorss,axs,0,show_text=False,linestyle='--')
        axs[0].scatter(x_moving_average[0], y_moving_average[0], marker="x", color='red')
        axs[0].set_aspect("equal", "box")
        axs[0].set_xlabel("X-coordinate (µm)")
        axs[0].set_ylabel("Y-coordinate (µm)")
        #axs[0].set_title("Trajectory")
            
            
            
        #plot_with_multicolor (np.arange(len(velocity_central1)) * gap1 * temporalScaling, velocity_central1, colorss,axs,1)
        for i in range(gap1,len(velocity_central1)-1 +gap1):
                
            #axs[1].plot(np.arange(len(velocity_central2))[i:i+2] * gap1 * temporalScaling, velocity_central2[i:i+2], label='Central Velocity',color=colorss[i], linestyle="-")
            axs[1].plot(np.arange(len(velocity_central1))[i:i+2] * gap1 * temporalScaling, velocity_central1[i:i+2], label='Central Velocity',color=colorss[i], linestyle="-")
            axs[2].scatter(angular_velocity_central1[i:i+2], velocity_central1[i:i+2], label='Central Velocity',color=colorss[i], s=3)
                
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Velocity (µm/s)")
        axs[2].set_xlabel("Angular Velocity  (rad/s)")
        axs[2].set_ylabel("Velocity (µm/s)")
        traj_vMean = np.mean(velocity_central1)
        comment = f'vMean={traj_vMean:.3f}'
        axs[1].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                 fontsize=8, ha='right', va='top')
        #creating Legend
        solid = mlines.Line2D([], [], color='black', linestyle='-', label='MA')
        dashed = mlines.Line2D([], [], color='black', linestyle='--', label='Backward Velocity')
        dotted = mlines.Line2D([], [], color='black', linestyle=':', label='Forward Velocity')
        dottedDash = mlines.Line2D([], [], color='black', linestyle='-.', label='MA')
        #axs[1].legend(handles=[solid, dashed, dotted, dottedDash])
        #axs[1].legend(handles=[solid])
            
        #plot_with_multicolor (np.arange(len(angular_velocity_central1)) * gap1 * temporalScaling, angular_velocity_central1, colorss,axs,2)
        for i in range(gap1,len(angular_velocity_central1) - 1 + gap1):
                
            #axs[2].plot(np.arange(len(angular_velocity_central2))[i:i+2] * gap1 * temporalScaling, angular_velocity_central2[i:i+2], label="ω  (Central)",color=colorss[i],linestyle="-")
            axs[3].plot(np.arange(len(angular_velocity_central1))[i:i+2] * gap1 * temporalScaling, angular_velocity_central1[i:i+2], label="ω  (Central)",color=colorss[i],linestyle="-")
            
        #creating Legend
        solid = mlines.Line2D([], [], color='black', linestyle='-', label='ω (MA)')
        dashed = mlines.Line2D([], [], color='black', linestyle='--', label='ω (Backward)')
        dotted = mlines.Line2D([], [], color='black', linestyle=':', label='ω (Forward)')
        dottedDash = mlines.Line2D([], [], color='black', linestyle='-.', label='ω (MA)')
        #axs[2].legend(handles=[solid, dashed, dotted, dottedDash])
        axs[3].legend(handles=[solid])
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Angular Velocity  (rad/s)")
        #axs[2].set_title("Angular Velocity")
        
        plt.suptitle(particle)
        plt.savefig(f"{dir_name}/{particle}.png",dpi=300)
        plt.tight_layout()
        plt.clf()
        plt.close(fig)
        plt.close('all')


# Calculate durations as trajectories belong to one bacteria for a long time
def shuffle_Durations(flat_list) :
    result = []
    temp_sum = 0
    for i in range(len(flat_list)):
        if isinstance(flat_list[i], tuple):
            temp_sum += flat_list[i][0]
            result.append(temp_sum)
            temp_sum = flat_list[i][1]
        else:
            temp_sum += flat_list[i]
            
    result.append(temp_sum)
    return result

def calculate_Histogram_Mean_STD (data, n_bin):

    # Calculate mean and standard deviation
    hist, bins = np.histogram(data, bins=n_bin, range=(0, 10), density=True)
    mean= np.mean(data)
    std_dev = np.std(data)
    bin_widths = np.diff(bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    weighted_mean = np.sum(hist * bin_centers * bin_widths)
    weighted_std = np.sqrt(sum(hist * (bin_centers - weighted_mean)**2 * bin_widths) / sum(hist * bin_widths))
    return mean, std_dev, weighted_mean, weighted_std
    
    

angV_threshold = 20
spatialScaling = 340.0/ 1280
temporalScaling = 1.0 / 30 # 30 frames per second
movingAverage_Iterator = 10
standingVelocity_Threshold = 2.0 # in µm/s
standingDuration_Threshold = 20 # in frames
mergingDistance_threshold = 0.0 # currently no merging happens
mergingFrame_threshold = 0
pairwiseDist_Threshold = 10.0 * spatialScaling
filterShortReverses = 0.3 #seconds

def bacteria_PostProcessor (tag) :
 
    #Initializations and inputs

    # Controlling parameters
    #tag = "ficoll0"
    directory = "."
    numInputFiles = 6
    generateTrajectories = True
    removeTrajectories = True
    largeWindowMovingAverage = False
    movingAverageGap = 7

    # Initializations
    velocity_data = []
    angVelocity_data = []
    durations = []
    runDurations = []
    runVelocities = []
    reverseCount = []
    trajDuration = []
    trajVelocity = []
    aveVelocities_after = []
    aveVelocities_before = []
    instVelocities_after = []
    instVelocities_before = []
    numberOfFilteredTrajectories = 0
    all_Particles = {}
    all_Particles_ID = []
    all_filteredParticles = {}
    turnAngle = []
    openBegin = []
    openEnd = []
    openBoth = []       #Trajectories with no reverse at all

    file_list = os.listdir(directory)
    # Create the tag directory if it doesn't exist and change to it
    if not os.path.exists(tag):
        os.makedirs(tag)

    numInputFiles = len([file for file in file_list if file.startswith(tag) and file.endswith(".txt")])
    for index in range(1, numInputFiles + 1):
        file_name = f"{tag}_{index}.txt"
        dir_name = f"{tag}/{tag}_{index}"
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        try:
            with open(file_name, "r") as input_file:
                data = input_file.readlines()
                
        except FileNotFoundError:
            print(f"File '{file_name}' not found. Skipping to the next file.")
            continue

        # Split data into separate particles
        particles = {}
        particles_ID = []
        for line in data:
            particle,angular_velocity ,velocity,time, x, y = line.strip().split()
            if particle not in particles:
                particles[particle] = []
            particles[particle].append((float(time), float(velocity), float(angular_velocity), float(x),float(y) ))
            
        # Convert the particle data into a Pandas dataframe
        particle_data_pandas = pd.DataFrame()
        particle_data_manual = []

         
        for particle, particle_data in particles.items():
            # Sort particle data by time
            particle_data = sorted(particle_data, key=lambda x: x[0])
            times = [d[0]* temporalScaling for d in particle_data]
            velocities = [d[1] * (spatialScaling / temporalScaling) for d in particle_data]
            angular_velocities = [d[2] * (1.0 /temporalScaling) for d in particle_data]
            x = [d[3] * spatialScaling for d in particle_data]
            y = [d[4] * spatialScaling for d in particle_data]
            gap1 = 1
            
            #calculating moving average with a fixed window or with small window but multiple times
            x_moving_average = moving_average(x, movingAverageGap)
            y_moving_average = moving_average(y, movingAverageGap)
            if largeWindowMovingAverage== False :
                tmpx = x
                tmpy = y
                for i in range(0,movingAverage_Iterator):
                    x_moving_average2 = moving_average(tmpx, 3)
                    y_moving_average2 = moving_average(tmpy, 3)
                
                    x_moving_average2 = np.concatenate(( [tmpx[0]],x_moving_average2, [tmpx[-1]] ) )
                    y_moving_average2 = np.concatenate( ([tmpy[0]], y_moving_average2,[tmpy[-1]] ) )
                    tmpx = x_moving_average2
                    tmpy = y_moving_average2

                x = x_moving_average2
                y = y_moving_average2
                #x_moving_average = x_moving_average2
                #y_moving_average = y_moving_average2
                movingAverageGap = 3
            
            # end of moving average calculations
            # Calculating velocities and other quantities
            velocity_central1 = calculate_velocity_central(x, y, temporalScaling, gap1)
            angular_velocity_central1 = calculate_angular_velocity_central(x, y, temporalScaling, gap1 )
            velocity_central2 = calculate_velocity_central(x_moving_average, y_moving_average, temporalScaling, gap1)
            angular_velocity_central2 = calculate_angular_velocity_central2(x_moving_average, y_moving_average, temporalScaling, gap1 )
            
            #adjusting the size of all lists
            events = find_events(angular_velocity_central1, 0)
            x,y,times,velocity_central1,angular_velocity_central1 = remove_data(x,y,times,velocity_central1,angular_velocity_central1,events,gap1,int((movingAverageGap-1)/2))
            
            events = find_events(angular_velocity_central2, 0)
            x_moving_average,y_moving_average,tmp,velocity_central2,angular_velocity_central2 = remove_data(x_moving_average,y_moving_average,times,velocity_central2,angular_velocity_central2,events,gap1,0)
            
            
            frames = [int( round( t / temporalScaling + 0.0001)) for t in times ]
            
            #Save everything in particles and particle_data_pandas.
            #particle_data_pandas is used for merging trajectories
            particle_data_updated = []
            for i in range(len(angular_velocity_central2)):
                particle_data_updated.append( (times[i],velocity_central1[i] ,angular_velocity_central1[i] ,x[i] , y[i],
                                                frames[i],velocity_central2[i],angular_velocity_central2[i], x_moving_average[i], y_moving_average[i]) )
                
            particles[particle] = particle_data_updated
            particles_ID.append(particle)
            particle_df = pd.DataFrame({
                'particle': particle,
                'frame': frames,
                'velocity': velocity_central1,
                'angular_velocity': angular_velocity_central1,
                'x': x,
                'y': y
            })
            #particle_data_pandas.append(particle_df)
            particle_data_pandas = pd.concat([particle_data_pandas, particle_df], ignore_index=True)
            #End of saving in particles and particle_data_pandas.
        
        # Filter trajectories that are not moving for a while
        if removeTrajectories == True :
            particles_copy = particles.copy()
            for particle, particle_data in particles_copy.items():
                #velocity_central2 = [d[6] for d in particle_data]
                velocity_central1 = [d[1] for d in particle_data]
                if max_consecutive_below_threshold(velocity_central1 , standingVelocity_Threshold) > standingDuration_Threshold :
                    new_key = str(index) + '_' + particle
                    all_filteredParticles[new_key] = particle_data
                    del particles[particle]
                    particles_ID.remove(particle)
                    numberOfFilteredTrajectories += 1
        
        
        #Handle merging
        grouped = particle_data_pandas.groupby('particle')

        merged_particles = []
        # Loop over all pairs of particle IDs (i, j)
        for i in range(len(particles_ID)):
            for j in range(i + 1, len(particles_ID)):
                prt1 = particles[particles_ID[i]]
                prt2 = particles[particles_ID[j]]
                #print(prt1)
                #print(prt2)
                # Check if the two particles meet the merging criteria
                if merge_trajectories(prt1, prt2, particles_ID[i], particles_ID[j], mergingDistance_threshold, mergingFrame_threshold):
                    merged_particles.append((particles_ID[i], particles_ID[j]))
                    #print(merged_particles)

        # Merge the particles in the `particles` list
        for pair in np.arange(len(merged_particles)-1,-1,-1) :
            #print(merged_particles[pair][0])
            prt1 = particles[merged_particles[pair][0]]
            prt2 = particles[merged_particles[pair][1]]
            merged_particle_data = prt1 + prt2[1:]
            #print( merged_particle_data)
            particles[merged_particles[pair][0]] = merged_particle_data
            del particles[merged_particles[pair][1]]
            particles_ID.index(merged_particles[pair][1])
            del particles_ID[ particles_ID.index(merged_particles[pair][1]) ]

        #handle plotting
        for particle, particle_data in particles.items():
        
            if len(x)>0 and find_max_pairwise_distance(x,y,pairwiseDist_Threshold) and generateTrajectories == True :
                plot_Trajectory_Panel(particle_data, particle, dir_name, gap1 )
        #plt.clf()
        #plt.close('all')

        # loop over all the particles in the particles dictionary and extract velocities
        #store all particle in all_particles
        for particle in particles:
            particle_data = particles[particle]
            # store all data with a unique key in a dictionary. The key is particle_index
            new_key = particle + '_' + str(index)
            all_Particles[new_key] = particle_data
            all_Particles_ID.append(particle)
            # append the velocity_central1 data for the current particle to the velocity_data list
            # (times,velocity_central1 ,angular_velocity_central1 ,x , y,
            # frames,velocity_central2,angular_velocity_central2, x_moving_average, y_moving_average)
            tmpV = []
            tmpAngV = []
            if largeWindowMovingAverage == True :
                velocity_data += [data[6] for data in particle_data]
                tmpV = np.array( [data[6] for data in particle_data] )
                angVelocity_data += [data[7] for data in particle_data]
                tmpAngV = np.array( [data[7] for data in particle_data] )
                tmpT = np.array( [data[0] for data in particle_data] )
            else :
            #use recursive moving average data short window
                velocity_data += [data[1] for data in particle_data]
                tmpV = np.array( [data[1] for data in particle_data] )
                angVelocity_data += [data[2] for data in particle_data]
                tmpAngV = np.array( [data[2] for data in particle_data] )
                tmpT = np.array( [data[0] for data in particle_data] )
            
            
            tmpFrame = np.array( [data[5] for data in particle_data] )
            events = find_events( tmpAngV , angV_threshold)
            
            if len(events) > 0:
                new_events = []
                curAngle = 0.0
                for i in range(len(events)-1,0,-1):
                    if events[i] - 1 == events[i-1]:
                        curAngle += tmpAngV[ events[i] ] * temporalScaling
                        continue
                    new_events.insert(0,events[i])
                    curAngle += tmpAngV[ events[i] ] * temporalScaling
                    turnAngle.append( curAngle)
                    curAngle = 0.0
                # Add the last element from my_list
                new_events.insert(0,events[0])
                curAngle += tmpAngV[ events[0] ] * temporalScaling
                turnAngle.append( curAngle)
                #print( new_events,'\t', events)
                events = new_events
            
            #this command exclude rapid runs
            #events = [events[i] for i in range(len(events)-1) if abs(events[i+1]-events[i]) >= filterShortReverses/ temporalScaling ]
            reverseCount.append( len(events) )
            trajDuration.append( (tmpT[-1] - tmpT[0]) )
            trajVelocity.append(np.mean(tmpV) )
            # calculate the duration of runs inclucing/excluding the very first and last ones
            if len(events) == 0:
                first_duration = len(tmpAngV)
                openBoth.append(first_duration * temporalScaling )
            else :
                first_duration = events[0]
                openBegin.append(first_duration * temporalScaling )
            #duration = first_duration - tmpFrame[0]
            duration = first_duration
            
            if duration > filterShortReverses / temporalScaling :
                durations.append(duration)
            for i in range(len(events)-1):
                duration = events[i+1] - events[i]
                tmpRunV = np.mean( tmpV[events[i]:events[i+1] ] )
                if duration > filterShortReverses / temporalScaling :
                    durations.append(duration)
                    runDurations.append(duration)
                    runVelocities.append( tmpRunV)
            if len(events) > 0:
                last_duration = len(tmpAngV)
                duration = last_duration - events[-1]
                openEnd.append(duration * temporalScaling)
                if duration > filterShortReverses / temporalScaling :
                    durations.append(duration)
                    
            # Loop over each event and find the average/instantaneous  velovities before and after each reverse
            for i, event in enumerate(events):
                aveV_after = []
                aveV_before = []
                v_before = []
                v_after = []
                
                if i < len(events) - 1 and i>0:
                # If this is not the last event, calculate velocities until next event
                    next_event = events[i+1]
                    for j in np.arange(event+1,next_event,1):
                        aveV_after.append (np.mean(tmpV[event:j]) )
                        v_after.append( tmpV[j] )
                        
                # If this is not the first event, calculate average velocities backward
                if i > 0 and i < len(events) - 1 :
                    prev_event = events[i-1]
                    for j in np.arange(event-1, prev_event, -1):
                        aveV_before.append( np.mean( tmpV[j:event]) )
                        v_before.append( tmpV[j] )
                        
                if len(aveV_after) > filterShortReverses*30 and len(aveV_before)> filterShortReverses*30:
                    aveVelocities_after.append (aveV_after.copy() )
                    aveVelocities_before.append (aveV_before.copy() )
                    instVelocities_after.append( v_after.copy() )
                    instVelocities_before.append( v_before.copy() )
            
                
    #Reshape the format of the lists if it's needed
    angVelocity_data = np.array(angVelocity_data)
    events = find_events(angVelocity_data, angV_threshold)
    #events = [events[i] for i in range(len(events)-1) if abs(events[i+1]-events[i]) >= filterShortReverses/ temporalScaling ]
    fRev = len(events)/len(angVelocity_data) * (1.0/ temporalScaling)
    nRev = len(events)
    vMean = np.mean(velocity_data)
    durations = np.array(durations)
    durations = durations * temporalScaling
    runDurations = np.array(runDurations)
    runDurations = runDurations * temporalScaling
    tMean = np.mean( durations)
    runTime_avg = np.mean( runDurations)
    reverseCount = np.array(reverseCount)
    runVelocities = np.array(runVelocities)
    #velocities_after = np.array(velocities_after)
    pairs_BegEnd = list(zip(openBegin, openEnd))
    allCuts = pairs_BegEnd
    allCuts.extend( openBoth)
    print( len(openBegin),len(openEnd),len(openBoth),len(allCuts))

    # Compute histogram and cumulative sum
    runDurationHist, bins = np.histogram(runDurations, bins=20, weights=np.ones(len(runDurations)) / len(runDurations) )
    cumulativeRunDurations = np.cumsum(runDurationHist[::-1])[::-1]

    # plot all trajectories in one figure
    endPointsX_shifted = []
    endPointsY_shifted = []
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('tab20' )
    for index, (particle, particle_data) in enumerate(all_Particles.items()):

        times = [d[0] for d in particle_data]
        x = [d[3] for d in particle_data] #d[8] for x_moving_average
        y = [d[4] for d in particle_data] #d[9] for x_moving_average
        
        # Shift the trajectories to start at (0, 0)
        x_shifted =x - x[0]
        y_shifted = y - y[0]
        endPointsX_shifted.append(x_shifted[-1] )
        endPointsY_shifted.append(y_shifted[-1] )
        color = cmap(index % cmap.N)
        ax.plot(x_shifted, y_shifted,color=color)
        #ax.plot(x_shifted, y_shifted)
        
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X-coordinate (µm)")
    ax.set_ylabel("Y-coordinate (µm)")
    comment = f'#Trajectories={len(all_Particles):.0f}'
    ax.annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=8, ha='right', va='top')
    
    plt.savefig(f"{dir_name}/Trajectories{tag}.png",dpi=300)
    plt.clf()
    plt.close(fig)
    plt.close('all')

    # create a figure with multiple subplots( mostly histograms)
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    axs[0,0].hist(velocity_data, bins=20,weights=np.ones(len(velocity_data)) / len(velocity_data) )
    axs[0,0].set_xlabel('Instantaneous Velocity (µm/s)')
    axs[0,0].set_ylabel('Percentage')
    axs[0,0].yaxis.set_major_formatter('{:.0%}'.format)
    comment = f'vMean={vMean:.3f}'
    axs[0,0].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top')

    axs[0,1].hist(angVelocity_data, bins=20, weights=np.ones(len(angVelocity_data)) / len(angVelocity_data) )
    axs[0,1].set_xlabel('MA ω (rad/s)')
    axs[0,1].set_ylabel('Percentage')
    axs[0,1].yaxis.set_major_formatter('{:.0%}'.format)

    comment = f'fRev={fRev:.3f}'
    axs[0,1].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top')
                     
    axs[0,2].hist(durations, weights=np.ones(len(durations)) / len(durations), bins=20)
    axs[0,2].set_xlabel('MA durations (s)')
    axs[0,2].set_ylabel('Percentage')
    axs[0,2].yaxis.set_major_formatter('{:.0%}'.format)
    #axs[0,2].yaxis.set_major_formatter(PercentFormatter(1))
    comment = f'tMean={tMean:.3f}'
    axs[0,2].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top')
                     
    axs[1,0].hist(runVelocities, weights=np.ones(len(runVelocities)) / len(runVelocities), bins=20)
    axs[1,0].set_xlabel('average run velocities')
    axs[1,0].set_ylabel('Percentage')
    axs[1,0].locator_params(axis='x', nbins=8)

    #axs[1,0].yaxis.set_major_formatter(PercentFormatter(1))
    axs[1,0].yaxis.set_major_formatter('{:.0%}'.format)
    comment = f'#Rev={nRev:.0f}'
    axs[1,0].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top')
    #comment = f'run_velocities = {runTime_avg:.3f}'
    #axs[1,0].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top')
                     
                     
    axs[1,1].hist(reverseCount, weights=np.ones(len(reverseCount)) / len(reverseCount), bins=20)
    axs[1,1].set_xlabel('Trajectories with # reverses')
    axs[1,1].set_ylabel('Percentage')
    axs[1,1].yaxis.set_major_formatter(PercentFormatter(1))
    comment = f'#Trajectories={len(all_Particles):.0f}'
    axs[1,1].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top')
    #axs[1,1].locator_params(axis='x', nbins=8)

    axs[1,2].hist(runDurations, weights=np.ones(len(runDurations)) / len(runDurations), bins=20)
    axs[1,2].set_xlabel('MA run durations (s)')
    axs[1,2].set_ylabel('Percentage')
    axs[1,2].yaxis.set_major_formatter(PercentFormatter(1))
    comment = f'run_mean = {runTime_avg:.3f}'
    axs[1,2].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top')
                     
    # Required code if you are willing to merge some subplots
    #merge [2,0] and [2,1]
    #axs[2,0].remove()
    #axs[2,1].remove()
    #gs = gridspec.GridSpec(3, 3, figure=fig)
    #ax_combined = plt.subplot(gs[6:8])

    #Plot the histogram of relative final location of bacteria with respect to the initial position
    # Compute the angle of each point with respect to the x-axis
    angles = np.arctan2(endPointsY_shifted, endPointsX_shifted)
    angles[angles < 0] += 2 * np.pi

    # Divide the circle into 8 sectors
    bin_edges = np.linspace(0, 2*np.pi, 9)
    counts, _ = np.histogram(angles, bins=bin_edges)
    counts = counts / len(endPointsX_shifted)

    # Plot the histogram
    labels = ['0-45', '45-90', '90-135', '135-180', '180-225', '225-270', '270-315', '315-360']
    axs[2,0].bar(labels, counts)
    axs[2, 0].set_xticks(range(len(labels)))
    axs[2,0].set_xticklabels(labels, rotation=90)
    axs[2,0].set_xlabel('Angle (degrees)')
    axs[2,0].set_ylabel('Relative final location')


    # histogram of turning angles of the reverses( events).
    #Remembrer that there is an event if angular velocity is high. Therefore no data for low turning angles
    axs[2,1].hist(turnAngle, weights=np.ones(len(turnAngle)) / len(turnAngle), bins=20)
    axs[2,1].set_xlabel('Turn Angle (rad)')
    axs[2,1].set_ylabel('Percentage')
    axs[2,1].yaxis.set_major_formatter('{:.0%}'.format)
    axs[2,1].set_xlim([np.pi *(1.0/8.0), np.pi*(1.1)])

    axs[2,2].plot(bins[:-1], cumulativeRunDurations, 'r--', label='Forward Accumulation')
    axs[2,2].bar (bins[:-1], cumulativeRunDurations, width=np.diff(bins), align='edge', label='Forward Accumulation')
    axs[2,2].set_xlabel('run Durations')
    axs[2,2].set_ylabel('Sojourn Probability')

    # adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(f"{dir_name}/histograms{tag}.png",dpi=300)
    plt.clf()
    plt.close(fig)
    plt.close('all')


    #plot instantaneous and average velocities before and after events
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    #cmap = plt.cm.get_cmap('viridis', len(instVelocities_before) )
    cmap = plt.cm.get_cmap('tab20' )
    for i, row in enumerate(instVelocities_before):
        if row:  # check if row is not empty
            #color = cmap(i)
            color = cmap(i % cmap.N)
            x = [-x*temporalScaling for x in range(len(row))]
            axs[0].plot(x, row, alpha=0.75,color=color)
            axs[0].text(x[-1], row[-1], str(i), va='center', ha='left', fontsize=3)
            
    for i, row in enumerate(instVelocities_after):
            if row:  # check if row is not empty
                #color = cmap(i)
                color = cmap(i % cmap.N)
                x = [x*temporalScaling for x in range(len(row))]
                axs[0].plot(x, row, alpha=0.75,color=color)
                axs[0].text(x[-1], row[-1], str(i), va='center', ha='left', fontsize=3)
                #print(len(x),len(row),x[0],row)
    ticks = axs[0].get_xticks()
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(['{}'.format(int(tick)) for tick in ticks])
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Instantaneous Velocity (µm/s)")



    for i, row in enumerate(aveVelocities_before):
        if row:  # check if row is not empty
            #color = cmap(i)
            color = cmap(i % cmap.N)
            x = [-x*temporalScaling for x in range(len(row))]
            axs[1].plot(x, row, alpha=0.75,color=color)
            axs[1].text(x[-1], row[-1], str(i), va='center', ha='left', fontsize=3)
            #print(len(x),len(row),x[0],row)
            
    for i, row in enumerate(aveVelocities_after):
            if row:  # check if row is not empty
                #color = cmap(i)
                color = cmap(i % cmap.N)
                x = [x*temporalScaling for x in range(len(row))]
                axs[1].plot(x, row, alpha=0.75,color=color)
                axs[1].text(x[-1], row[-1], str(i), va='center', ha='left', fontsize=3)
                #print(len(x),len(row),x[0],row)
                
    ticks = axs[1].get_xticks()
    axs[1].set_xticks(ticks)
    #axs[1].set_xticklabels(['{}'.format(int(tick)) for tick in ticks])
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Average Velocity (µm/s)")

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(f"{dir_name}/Events{tag}.png",dpi=300)
    plt.clf()
    plt.close(fig)
    plt.close('all')
    #End of plot: instantaneous and average velocities before & after events


    # plot filtered trajectories and save them in the related folder
    directory = f"{tag}/{tag}_FilteredTrajectories"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for particle, particle_data in all_filteredParticles.items():
        if generateTrajectories == True :
            plot_Trajectory_Panel(particle_data, particle, directory, gap1)
    plt.clf()
    plt.close('all')

    print(len(all_Particles),numberOfFilteredTrajectories)

    # Shuffle durations and plot the histogram
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle('Histograms of Shuffled Durations', fontsize=24)
    allCuts_copy = allCuts.copy()
    for i in range(5):
        for j in range(5):
            # Shuffle the list
            random.shuffle(allCuts_copy)
            result = shuffle_Durations (allCuts_copy)
            result.extend(runDurations)
            
            # Plot the histogram with frequency instead of counts
            axs[i,j].hist(result, weights=np.ones(len(result)) / len(result), bins=20,range=(0, 10) )
            #axs[i,j].set_ylim ([0,1] )
            #convert frequency to percentage
            axs[i,j].yaxis.set_major_formatter(PercentFormatter(1))
            
            
            mean, std_dev, weighted_mean  , weighted_std  = calculate_Histogram_Mean_STD( result, 20)

            axs[i,j].annotate(r'$\mu=%.2f$' % mean, xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=8, ha='right', va='top')
            axs[i,j].annotate(r'$\sigma=%.2f$' % std_dev, xy=(0.95, 0.85), xycoords='axes fraction',
                fontsize=8, ha='right', va='top')
            axs[i,j].annotate(r'$\mu_w=%.2f$' % weighted_mean, xy=(0.95, 0.75), xycoords='axes fraction', fontsize=8, ha='right', va='top')
            axs[i,j].annotate(r'$\sigma_w=%.2f$' % weighted_std, xy=(0.95, 0.65), xycoords='axes fraction', fontsize=8, ha='right', va='top')
            axs[i,j].annotate(r'$\#=%.0f$' % len(result), xy=(0.95, 0.55), xycoords='axes fraction', fontsize=8, ha='right', va='top')
            
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(f"{dir_name}/Shuffle{tag}.png",dpi=300)
    plt.clf()
    plt.close(fig)
    plt.close('all')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    #fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('tab20' )
    for index, (particle, particle_data) in enumerate(all_Particles.items()):

        times = [d[0] for d in particle_data]
        v = [d[1] for d in particle_data] #d[8] for x_moving_average
        w = [d[2] for d in particle_data] #d[9] for x_moving_average
        color = cmap(index % cmap.N)
        axs[0,0].scatter(w, v,color=color, s=2, alpha=0.5,edgecolors='none')
        
    #ax.set_aspect("equal", "box")
    axs[0,0].set_xlabel("Angular Velocity (rad/s)",fontsize = 14)
    axs[0,0].set_ylabel("Velocity (µm/s)", fontsize = 14)
    comment = f'#Trajectories={len(all_Particles):.0f}'
    axs[0,0].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=14, ha='right', va='top')
                
                
    #for i in range(len(reverseCount) ):
    cmap = plt.cm.get_cmap('tab20' )
    indices = np.arange(len(endPointsX_shifted))
    color = cmap(np.linspace(0,1,len(reverseCount) ) )
    axs[0,1].scatter(trajDuration,reverseCount,c=indices,cmap=cmap, s=10, alpha=1.0,edgecolors='none')
    axs[0,1].set_xlabel("Trajectory Duration (s)",fontsize = 14)
    axs[0,1].set_ylabel("# Reverses",fontsize = 14)

    axs[1,1].scatter(trajVelocity,reverseCount,c=indices,cmap=cmap, s=10, alpha=1.0,edgecolors='none')
    axs[1,1].set_xlabel("Trajectory Velocity (µm/s)",fontsize = 14)
    axs[1,1].set_ylabel("# Reverses",fontsize = 14)

    # Scatter plot of final locations in one figure
    msd = sum(xi**2 + yi**2 for xi, yi in zip(endPointsX_shifted, endPointsY_shifted) )/ len(endPointsX_shifted)
    axs[1,0].scatter(endPointsX_shifted, endPointsY_shifted , c=indices ,cmap=cmap, s=10,edgecolors='none')
    axs[1,0].set_aspect("equal", "box")
    axs[1,0].set_xlabel("X-coordinate (µm)",fontsize = 14)
    axs[1,0].set_ylabel("Y-coordinate (µm)",fontsize = 14)
    comment = f'MSD={msd:.2f}'
    axs[1,0].annotate(comment, xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=14, ha='right', va='top')

    plt.savefig(f"{dir_name}/Scatters_{tag}.png",dpi=300)
    plt.clf()
    plt.close(fig)
    plt.close('all')
    return velocity_data, angVelocity_data, durations, runDurations,runVelocities, reverseCount, trajDuration, trajVelocity, all_Particles, all_Particles_ID, all_filteredParticles, turnAngle


class ProcessedData:
    def __init__(self,velocity_data, angVelocity_data, durations, runDurations,runVelocities, reverseCount, trajDuration, trajVelocity, all_Particles, all_Particles_ID, all_filteredParticles, turnAngle   ):
        self.velocity_data = velocity_data
        self.angVelocity_data = angVelocity_data
        self.durations = durations
        self.runDurations = runDurations
        self.runVelocities = runVelocities
        self.reverseCount = reverseCount
        self.trajDuration = trajDuration
        self.trajVelocity = trajVelocity
        #aveVelocities_after = aveVelocities_after
        #aveVelocities_before = aveVelocities_before
        #instVelocities_after = instVelocities_after
        #instVelocities_before = instVelocities_before
        #numberOfFilteredTrajectories = 0
        self.all_Particles = all_Particles
        self.all_Particles_ID = all_Particles_ID
        self.all_filteredParticles = all_filteredParticles
        self.turnAngle = turnAngle
        #openBegin = []
        #openEnd = []
        #openBoth = []

def parallel_Calculation(tag):
    tmpClass = ProcessedData(*bacteria_PostProcessor(tag))
    return (tag, tmpClass)
    
    
class_dict = {}
tags = ["ficoll10","ficoll15","ficoll20"]
#for tag in tags:
#   tmpClass = ProcessedData( *bacteria_PostProcessor(tag) )
#   class_dict[tag] = tmpClass

# Create a pool of processes
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    results = pool.map(parallel_Calculation, tags)
    class_dict = {tag: tmpClass for tag, tmpClass in results}
    pool.close()
    pool.join()

    #statistic, p_value = stats.ks_2samp(class_dict["ficoll5"].turnAngle, class_dict["ficoll25"].turnAngle)
    #print("KS Test Results:")
    #print("Statistic:", statistic)  #0.09168720488310489
    #print("p-value:", p_value) # 8.186354217440532e-26

