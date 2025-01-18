from multiprocessing import Process, Queue
import numpy as np
 
from mayavi import mlab
 
# Global map // 3D map visualization using pangolin
class Map(object):
    def __init__(self):
        self.frames = [] # camera frames [means camera pose]
        self.points = [] # 3D points of map
        self.state = None # variable to hold current state of the map and cam pose
        self.q = None # A queue for inter-process communication. | q for visualization process
 
    def create_viewer(self):
        # Parallel Execution: The main purpose of creating this process is to run 
        # the `viewer_thread` method in parallel with the main program. 
        # This allows the 3D viewer to update and render frames continuously 
        # without blocking the main execution flow.
         
        self.q = Queue() # q is initialized as a Queue
 
        # initializes the Parallel process with the `viewer_thread` function 
        # the arguments that the function takes is mentioned in the args var
        p = Process(target=self.viewer_thread, args=(self.q,)) 
         
        # daemon true means, exit when main program stops
        p.daemon = True
         
        # starts the process
        p.start()
 
    def viewer_thread(self, q):
        # `viewer_thread` takes the q as input
        # initializes the viz window
        self.viewer_init(1280, 720)
        # An infinite loop that continually refreshes the viewer
        while True:
            self.viewer_refresh(q)
 
    def viewer_init(self, w, h):
        #pangolin.CreateWindowAndBind('Main', w, h)
        mlab.figure(size=(w, h), bgcolor=(1, 1, 1))
        mlab.view(azimuth=90, elevation=90, distance=10)
 
        
 
    def viewer_refresh(self, q):
        # Checks if the current state is None or if the queue is not empty.
        if self.state is None or not q.empty():
            # Gets the latest state from the queue.
            self.state = q.get()

        mlab.clf()  # Clear the figure

        # Set background color to white (optional since mlab.figure already sets it).
        mlab.figure(bgcolor=(1, 1, 1))

        if self.state is not None:
            poses, pts = self.state

            # Draw camera trajectory as lines
            for pose in poses:
                mlab.points3d(pose[0], pose[1], pose[2], color=(0, 1, 0), scale_factor=0.1)

            # Draw points in the point cloud
            mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 0, 0), scale_factor=0.05)

        mlab.draw()  # Refresh the visualization
 
     
    def display(self):
        if self.q is None:
            return
        poses, pts = [], []
        for f in self.frames:
            # updating pose
            poses.append(f.pose)
 
        for p in self.points:
            # updating map points
            pts.append(p.pt)
         
        # updating queue
        self.q.put((np.array(poses), np.array(pts)))

class Point(object):
    # A Point is a 3-D point in the world
    # Each point is observed in multiple frames
 
    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []
 
        # assigns a unique ID to the point based on the current number of points in the map.
        self.id = len(mapp.points)
        # adds the point instance to the map’s list of points.
        mapp.points.append(self)
 
    def add_observation(self, frame, idx):
        # Frame is the frame class
        self.frames.append(frame)
        self.idxs.append(idx)