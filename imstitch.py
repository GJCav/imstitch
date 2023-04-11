#!/usr/bin/env python
# coding: utf-8

"""
The imstitch script is used to stitch images together according to their matched points.
Usage: type in your terminal
    cd <where you store the images>
    python3 imstitch.py
Then imstitch find all image files (jpg, png) in the current directory and stitch them together.
The result is saved as "stitched_image.jpg" in the current directory.

NOTE: the ALPHA channel in png file is not supported and discarded.

Copyright Information:
- Author: GJCav, micojcav@outlook.com
- Date: 2023-03-27 UTF+8
- License: MIT
"""

import cv2
import numpy as np
import os
import os.path as path
import sys
REQUIRE_MATCH_POINT = 10
MATCH_THRESHOLD = 0.2      # the smaller, the better


def show_image(cvimage):
    cv2.imshow('Stitched Image', cvimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stitch_two_images(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < MATCH_THRESHOLD*n.distance:
            good_matches.append([m])
            
    if len(good_matches) < REQUIRE_MATCH_POINT:
        raise Exception("not enough matched points")
        
    good_matches = [item for sublist in good_matches for item in sublist] # flatten the array good_matches
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    # result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    
    # avoid black pixel (null pixel originated from warping) overlapping pixel with meaningful color
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = np.maximum(result[t[1]:h1+t[1],t[0]:w1+t[0]], img1)
    return result


def preview_match(img1, img2, show=True):
    # Create a SIFT object to detect keypoints and compute descriptors
    sift = cv2.SIFT_create()
    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    # Create a Brute-Force Matcher object to match the descriptors
    bf = cv2.BFMatcher()
    # Match the descriptors between the images
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Apply ratio test to select good matches
    good_matches = []
    for m,n in matches:
        if m.distance < MATCH_THRESHOLD*n.distance:
            good_matches.append([m])
            
    # match preview
    if show:
        img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
        show_image(img3)
    
    return good_matches


def list_jpg_files():
    return [e for e in os.listdir() if path.isfile(e) and e.lower()[-3:] in ["jpg", "png"]]

if __name__ == "__main__":

    imgfile = list_jpg_files()


    if len(imgfile) <= 1:
        print("no need to stitch")
        sys.exit(0)


    imgs = {}
    for f in imgfile:
        imgs[f] = cv2.imread(f)


    # find Maximum Spanning Tree
    N = len(imgfile)
    edgs = []
    for i in range(N):
        for j in range(i+1, N):
            cnt = len(preview_match(imgs[imgfile[i]], imgs[imgfile[j]], show=False))
            if cnt > REQUIRE_MATCH_POINT:
                edgs.append((i, j, cnt))
    edgs.sort(key=lambda x: x[2], reverse=True)


    class UnionSet:
        def __init__(self, N):
            self.fa = list(range(N))
            
        def find(self, u):
            fa = self.fa
            
            t = u
            v = fa[t]
            while v != t:
                t = v
                v = fa[v]
            
            root = v
            v = fa[u]
            while v != u:
                fa[u] = root
                u = v
                v = fa[v]
                
            return root
                
        
        def connect(self, u, v):
            u = self.find(u)
            v = self.find(v)
            if u == v: return
            self.fa[u] = v
            
        def connected(self, u, v):
            return self.find(u) == self.find(v)


    s = UnionSet(N)
    tree_edgs = []
    for e in edgs:
        if len(tree_edgs) == N-1: break
        u, v, _ = e
        if s.connected(u, v): continue
        s.connect(u, v)
        tree_edgs.append(e)
        
    if len(tree_edgs) < N - 1:
        print("program can't find a proper way to stitch these images")
        print("stitch information are shown as follow")
        fa = s.fa
        for i in range(N):
            s.find(i)        # flatten the tree
        for i in range(N):
            if fa[i] != i: continue # not group root
            member = []
            for j in range(N):
                if fa[j] == i: member.append(imgfile[j])
            print("- " + ", ".join(member))
        sys.exit(0)


    # walk the tree and stitch the images
    vis = [0] * N
    print(f"stitch image {imgfile[ tree_edgs[0][0] ]}")
    rst = imgs[imgfile[ tree_edgs[0][0] ]]
    vis[tree_edgs[0][0]] = 1
    def walk(h):
        global vis, rst
        
        for e in tree_edgs:
            to = None
            if e[0] == h: to = e[1]
            elif e[1] == h: to = e[0]
            else: continue
            
            if vis[to]: continue
            
            print(f"stitch image {imgfile[to]}")
            rst = stitch_two_images(rst, imgs[imgfile[to]])
            vis[to] = True
            walk(to)
    walk(tree_edgs[0][0])

    # show_image(rst)
    cv2.imwrite("stitched_image.jpg", rst)



