import numpy as np
from math import ceil
import math

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

class Hough:
    def __init__(self, n_classes, image_WH):
        self.n_classes = n_classes
        self.image_WH = image_WH
        self.img_classes = np.zeros([n_classes, image_WH, image_WH])
        self.depth_classes = np.zeros([n_classes])
        self.roi_classes_minx = np.full([n_classes], image_WH)
        self.roi_classes_maxx = np.full([n_classes], 0) 
        self.roi_classes_miny = np.full([n_classes], image_WH) 
        self.roi_classes_maxy = np.full([n_classes], 0) 
        self.pts_classes = [[] for _ in range(n_classes)]

    def cast_votes(self, labels, directionsx, directionsy, distancez):
        for x in range(self.image_WH):
            for y in range(self.image_WH):
                self.pts_classes[labels[x][y]].append( (x, y) )
                self.roi_classes_minx[labels[x][y]] = min(x, self.roi_classes_minx[labels[x][y]])
                self.roi_classes_maxx[labels[x][y]] = max(x, self.roi_classes_maxx[labels[x][y]])
                self.roi_classes_miny[labels[x][y]] = min(y, self.roi_classes_miny[labels[x][y]])
                self.roi_classes_maxy[labels[x][y]] = max(y, self.roi_classes_maxy[labels[x][y]])
                if labels[x][y] != 0:
                    self.incre_pixels_along_line( labels[x][y], labels, directionsx[x][y], directionsy[x][y], (x, y) )
                    self.depth_classes[labels[x][y]] += distancez[x][y]
    
    def incre_pixels_along_line(self, class_no, labels, unitx, unity, point):
        if np.isnan(unitx) or np.isnan(unity):
            return
        if unitx == 0 and unity == 0:
            self.img_classes[class_no][point[0]][point[1]] += 1
            return
        x1, y1  = self.locate_endpoint(point[0], point[1], unitx, unity)
        pts = self.bresenham_line(point[0], point[1], x1, y1)
        for p in pts:
            if labels[int(p[0])][int(p[1])] != 0:
                self.img_classes[class_no][int(p[0])][int(p[1])] += 1
    
    def tally_votes(self):
        results = []
        for c in range(1, self.n_classes):
            x, y = np.unravel_index(np.argmax(self.img_classes[c]), self.img_classes[c].shape)
            if len(self.pts_classes[c]) > 0:
                z = self.depth_classes[c] / len(self.pts_classes[c])
            elif not np.isnan(self.depth_classes[c]):
                z = self.depth_classes[c]
            else:
                z = 0
            results.append( (x, y, z) )
        return results
    
    def get_rois(self):
        rois = []
        for c in range(1, self.n_classes):
            roi = self.calculate_roi(c)
            rois.append(roi)
        return rois

    def locate_endpoint(self, x0, y0, unitx, unity):
        # Infinite slope handling
        if unitx != 0:
            #y = (unity / unitx) * x + y0
            yp1 = int((unity / unitx) * 0 + y0)
            if yp1 >= 0 and yp1 < self.image_WH:
                if unitx < 0:
                    return 0, yp1
            yp1 = int((unity / unitx) * (self.image_WH - 1) + y0)
            if yp1 >= 0 and yp1 < self.image_WH:
                if unitx > 0:
                    return self.image_WH - 1, yp1
        if unity != 0:
            #(yp1 - y0) * (unitx / unity) = xp1
            xp1 = int((unitx / unity)* (0 - y0))
            if xp1 >= 0 and xp1 < self.image_WH:
                if unity < 0:
                    return xp1, 0
            xp1 = int((unitx / unity)* (self.image_WH - 1 - y0))
            if xp1 >= 0 and xp1 < self.image_WH:
                if unity > 0:
                    return xp1, self.image_WH - 1
        
    def bresenham_line(self, x0, y0, x1, y1):
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0  
            x1, y1 = y1, x1
        switched = False
        if x0 > x1:
            switched = True
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        if y0 < y1: 
            ystep = 1
        else:
            ystep = -1
        deltax = x1 - x0
        deltay = abs(y1 - y0)
        error = -deltax / 2
        y = y0
        line = []    
        for x in range(x0, x1 + 1):
            if steep:
                line.append((y,x))
            else:
                line.append((x,y))
            error = error + deltay
            if error > 0:
                y = y + ystep
                error = error - deltax
        if switched:
            line.reverse()
        return line
    
    def calculate_roi(self, class_no):
        minx = self.roi_classes_minx[class_no]
        maxx = self.roi_classes_maxx[class_no]
        miny = self.roi_classes_miny[class_no]
        maxy = self.roi_classes_maxy[class_no]
        return [minx / self.image_WH, miny / self.image_WH, maxx / self.image_WH, maxy / self.image_WH]
        