from queuelib import *
from stacklib import *


class graph:
    def __init__(self):
        graph = []

    def addVertex(self, n):
        if n < 0:
            return -1
        for i in range(0, n):
            self.graph.append([])
        return len(self.graph)

    def addEdge(self, from_idx, to_idx, directed, weight):
        if weight == 0:
            return False
        self.graph[from_idx].append([to_idx, weight])
        if directed is False:
            self.graph[to_idx].append([from_idx, weight])
        return True

    def traverse(self, start, typeBreadth):
        tempLOT = []
        if start is not None:
            print start
            tempLOT.append(start)
            for i in self.graph[start]:
                print i[0]
                tempLOT.append(i[0])
        if typeBreadth is True:
            x = queue()
        elif typeBreadth is False:
            x = stack()
        dis = [False for i in len(self.graph)]
        proc = [False for j in len(self.graph)]
        for k in len(self.graph):
            if dis[k] is False:
                x.push(k)
                dis[k] = True
            while len(x.store) != 0:
                p = x.pop()
                if proc[p] is False:
                    print p
                    tempLOT.append(p)
                    proc[p] = True
                for l in self.graph[p]:
                    if dis[l[0]] is False:
                        x.push(l[0])
                        dis[l[0]] = True

    def connectivity(self, vx, vy):
        rval = [False for i in range(2)]
        for j in self.graph[vx]:
            if j[0] == vy:
                rval[0] = True
        for k in self.graph[vy]:
            if k[0] == vx:
                rval[1] = True
        return rval

    def path(self, vx, vy):
        tempLOT = []
        x = queue()
        dis = [False for i in len(self.graph)]
        proc = [False for j in len(self.graph)]
        if dis[vx] is False:
            x.push(vx)
            dis[vx] = True
        while len(x.store) != 0:
            p = x.pop()
            if proc[p] is False:
                print p
                tempLOT.append(p)
                proc[p] = True
            for l in self.graph[p]:
                if dis[l[0]] is False:
                    x.push(l[0])
                    dis[l[0]] = True
        found = [False for i in range(2)]
        for i in tempLOT:
            if i == vy:
                found[0] = tempLOT
