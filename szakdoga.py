import cv2
import numpy as np
import triangle as tri
from shapely.geometry import Polygon, asPolygon

depthWindow = "depth"
intensityWindow = "intensity"
blendedWindow = "blended"

depthTransformPoints = np.empty((3,2),dtype=np.float32)
lastAppended=0
rectList = []


def is_first_rect_inside_second(first_rect, second_rect):
    rect_project_to_points = np.empty(4, dtype=np.bool)

    for project_from_index in range(0, 4):
        project_to_sides = np.empty(4, dtype=np.bool) # true if point is on the positive side of side

        for project_to_side_index in range(0, 3):
            project_to_sides[project_to_side_index] = 0 > test_which_side(
                first_rect[project_to_side_index],
                first_rect[project_to_side_index + 1],
                second_rect[project_from_index])

        project_to_sides[3] = 0 > test_which_side(
            first_rect[3],
            first_rect[0],
            second_rect[project_from_index])

        rect_project_to_points[project_from_index] = project_to_sides.all()

    if rect_project_to_points.all():
        return 1
    if not rect_project_to_points.any():
        return -1

    return 0


def triangle_is_in_set(triangle,forbiddet_segment_set):

    sub_test_1 = (triangle[0], triangle[1]) in forbiddet_segment_set or (triangle[1], triangle[0]) in forbiddet_segment_set
    sub_test_2 = (triangle[0], triangle[2]) in forbiddet_segment_set or (triangle[2], triangle[0]) in forbiddet_segment_set
    sub_test_3 = (triangle[1], triangle[2]) in forbiddet_segment_set or (triangle[2], triangle[1]) in forbiddet_segment_set

    return (sub_test_1 and sub_test_2 and sub_test_3)


def test_which_side(point1, point2, point_under_test):

    side_vector = point2-point1
    side_vector2 = np.empty_like(side_vector)
    side_vector2[0] = side_vector[1]
    side_vector2[1] = -1*side_vector[0]
    to_test_point_vector = point1-point_under_test

    return side_vector2.dot(to_test_point_vector)


def make_obj_file(geometry_filename, vertices, faces, texture_coordinates, material_filename, mat_to_use):
    geometry_file = open(geometry_filename, "w")
    material_file = open(material_filename, "w")

    geometry_file.write("mtllib %s \n" % material_filename)
    for v in vertices:
        geometry_file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))

    for vt in texture_coordinates:
        geometry_file.write("vt %.4f %.4f\n" % (vt[1], vt[0]))

    geometry_file.write("usemtl %s\n" % mat_to_use)
    for face in faces:
        geometry_file.write("f %d/%d %d/%d %d/%d\n" % (face[0],face[0], face[1],face[1], face[2],face[2]))

    material_file.write("newmtl %s\n" % mat_to_use)
    material_file.write("Ka 1.0 1.0 1.0\n")
    material_file.write("Kd 1.0 1.0 1.0\n")
    material_file.write("Ks 0.0 0.0 0.0\n")
    material_file.write("d 1.0v")
    material_file.write("illum 2\n")
    material_file.write("map_Ka texture.jpg\n")
    material_file.write("map_Kd texture.jpg\n")

def nothing(x):
    pass

class MyRectangle:
    def __init__(self):
        self.points = np.empty((4,3),dtype=np.float32)
        self.pointsAdded = 0
        self.beta = np.empty((1,3))

    def addPoint(self,p):
        self.points[self.pointsAdded] = p
        self.pointsAdded = self.pointsAdded + 1

    def number_of_added_points(self):
        return self.pointsAdded

    def regression_coeffitients(self):
        X = np.empty((4,3))
        X[:,1:3] = self.points[:,0:2]
        X[:,0] = 1
        y = self.points[:,2]

        Xtranspose = np.transpose(X)
        y = y.transpose()

        beta = np.linalg.inv(Xtranspose.dot(X)).dot(Xtranspose).dot(y)

        self.beta = beta

        return beta

    def estimate_point(self,point):
        pt = np.array((1, point[0], point[1]))

        z = pt.dot(self.beta)

        return np.array((point[0], point[1], z))

def mouseCallbackDepth(event,x,y,flags,param):
    global lastAppended

    if event == cv2.EVENT_LBUTTONDBLCLK and lastAppended < 3:
        print([x, y])
        depthTransformPoints[lastAppended, 0] = x
        depthTransformPoints[lastAppended, 1] = y
        lastAppended = lastAppended+1


def mouseCallbackBlended(event,x,y,flags,param):

    if(event == cv2.EVENT_LBUTTONDBLCLK):

        point_to_add = [y, x, param[y, x]]
        print("added: ",point_to_add)
        if len(rectList) == 0:

            rectList.append(MyRectangle())
            rectList[0].addPoint(point_to_add)

        else:

            lastElement = len(rectList)-1
            if rectList[lastElement].number_of_added_points() < 4:
                rectList[lastElement].addPoint(point_to_add)
            else:
                rectList.append(MyRectangle())
                rectList[lastElement+1].addPoint(point_to_add)

def main():

    depth_img = cv2.imread("./images/depth/kaula_depth_map_50fov.png",cv2.IMREAD_GRAYSCALE)
    intensity_img = cv2.imread("./images/intensity/kaula_int_4.jpg",cv2.IMREAD_GRAYSCALE)

    depth_img_transformed = np.empty_like(intensity_img)

    cv2.namedWindow(depthWindow)
    cv2.namedWindow(intensityWindow)
    cv2.namedWindow(blendedWindow, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("contour", cv2.WINDOW_AUTOSIZE)

    depth_img = cv2.pyrDown(depth_img)

    intensity_img = cv2.pyrDown(intensity_img)
    intensity_img = cv2.pyrDown(intensity_img)
    intensity_img = cv2.pyrDown(intensity_img)

    cv2.imshow(depthWindow, depth_img)
    cv2.imshow(intensityWindow, intensity_img)

    cv2.waitKey(1000)

    cv2.createTrackbar("lowerThresh", depthWindow, 5, 255, nothing)
    cv2.createTrackbar("higherThresh", depthWindow, 8, 255, nothing)
    cv2.createTrackbar("lowerThresh", intensityWindow, 5, 255, nothing)
    cv2.createTrackbar("higherThresh", intensityWindow, 10, 255, nothing)
    cv2.createTrackbar("corner offset", intensityWindow, 5, 255, nothing)
    cv2.createTrackbar("bilat int", intensityWindow, 50, 255, nothing)
    cv2.createTrackbar("bilat space", intensityWindow, 50, 255, nothing)
    cv2.createTrackbar("blend", blendedWindow, 5, 255, nothing)

    contours_found = []

    cv2.setMouseCallback(depthWindow,mouseCallbackDepth)

    pressedKey = cv2.waitKey(100)

    while pressedKey != 27:

        if pressedKey == ord('c'): #key c

            lower_threshold = cv2.getTrackbarPos("lowerThresh", depthWindow)
            higher_threshold = cv2.getTrackbarPos("higherThresh", depthWindow)
            depth_img_edge = cv2.Canny(depth_img, lower_threshold, higher_threshold)

            bilat_int = cv2.getTrackbarPos("bilat int", intensityWindow)
            bilat_col = cv2.getTrackbarPos("bilat space", intensityWindow)

            intensity_img_filtered = cv2.bilateralFilter(intensity_img, 7, bilat_int, bilat_col)
            lower_threshold = cv2.getTrackbarPos("lowerThresh", intensityWindow)
            higher_threshold = cv2.getTrackbarPos("higherThresh", intensityWindow)
            intensity_img_edge = cv2.Canny(intensity_img_filtered, lower_threshold, higher_threshold)

            structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            cv2.dilate(intensity_img_edge, structuring_element, intensity_img_edge)
            cv2.erode(intensity_img_edge, structuring_element, intensity_img_edge)

            corner_offset = cv2.getTrackbarPos("corner offset", intensityWindow)

            corners = cv2.goodFeaturesToTrack(intensity_img, 10, 0.02, 100)

            for i,point in zip(range(0,3),corners[corner_offset:corner_offset+3]):
                cvt_ponint= (int(point[0,0]), int(point[0,1]))
                cv2.circle(intensity_img_edge,cvt_ponint,5,(130+i*50),2)

            cv2.imshow(depthWindow, depth_img_edge)
            cv2.imshow(intensityWindow, intensity_img_edge)

        if pressedKey == ord('t'):  # key t

            transformMatrix = cv2.getAffineTransform(corners[corner_offset:corner_offset+3],depthTransformPoints)
            #transformMatrix = cv2.getAffineTransform(corners[corner_offset:corner_offset+3],np.array([[286, 13],[107, 15],[338, 88]],dtype=np.float32))

            map_x = np.empty_like(intensity_img, dtype=np.float32)
            map_y = np.empty_like(intensity_img, dtype=np.float32)

            for x in range(intensity_img.shape[0]):
                for y in range(intensity_img.shape[1]):

                    transformed = np.dot(transformMatrix,[y, x, 1])
                    map_x[x, y] = transformed[0]
                    map_y[x, y] = transformed[1]

            depth_img_transformed = cv2.remap(depth_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            cv2.setMouseCallback(blendedWindow, mouseCallbackBlended, param=depth_img_transformed)

            intensity_img_edge[0, :] = 255
            intensity_img_edge[-1, :] = 255
            intensity_img_edge[:, 0] = 255
            intensity_img_edge[:, -1] = 255

            # lower_threshold = cv2.getTrackbarPos("lowerThresh", depthWindow)
            # higher_threshold = cv2.getTrackbarPos("higherThresh", depthWindow)
            contour = np.copy(intensity_img_edge)

            image, contours_found, hierarchy = cv2.findContours(contour,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            print("contour shape",contour.shape)
            cv2.imshow(depthWindow,contour)
            print("transformed depth size:", depth_img_transformed.shape, "intensity img size", intensity_img.shape)

        if pressedKey == ord('b'):
            blend = cv2.getTrackbarPos("blend", blendedWindow)
            blended_img = cv2.addWeighted(depth_img_transformed, blend / 255.0, intensity_img, 1 - (blend / 255.0), 0.0)

            for rect in rectList:
                number_of_pts = rect.number_of_added_points()
                for i in range(1,number_of_pts):
                    cv2.line(blended_img,(rect.points[i,1], rect.points[i,0]),(rect.points[i-1,1], rect.points[i-1,0]), 0, 2)
                if number_of_pts == 4:
                    cv2.line(blended_img, (rect.points[3, 1], rect.points[3, 0]), (rect.points[0, 1], rect.points[0, 0]), 0, 2)


            cv2.imshow(blendedWindow, blended_img)

        if pressedKey == ord('p'):

            print("model writing started")

            globalVertexList = np.empty((1,3),dtype=np.float32) # first element is empty

            for rect in rectList:
                rect.regression_coeffitients()

            rectList.sort(key=lambda rect: rect.beta[0]) # ezt lehet meg kel forditani

            faceList = []

            for i in range(len(rectList)): # goes through every rect

                localVertexList = np.empty((4,3), dtype=np.float32)
                localSegmentList = np.empty((1,2), dtype=np.int)
                local_rect_list = []

                localVertexList[0, :] = rectList[i].estimate_point(rectList[i].points[0, :2])
                localVertexList[1, :] = rectList[i].estimate_point(rectList[i].points[1, :2])
                localVertexList[2, :] = rectList[i].estimate_point(rectList[i].points[2, :2])
                localVertexList[3, :] = rectList[i].estimate_point(rectList[i].points[3, :2])

                for j in reversed( range(i) ): #select every rectangle before current

                    rect_project_to = asPolygon(rectList[i].points[:, :2])
                    projected_rect = asPolygon(rectList[j].points[:, :2])

                    inside = projected_rect.within(rect_project_to)

                    if inside == True:

                        segLen = len(localVertexList)

                        temp_vertex_list = np.empty((4, 3), dtype=np.float32)

                        temp_vertex_list[0] = rectList[i].estimate_point(rectList[j].points[0, :2])
                        temp_vertex_list[1] = rectList[i].estimate_point(rectList[j].points[1, :2])
                        temp_vertex_list[2] = rectList[i].estimate_point(rectList[j].points[2, :2])
                        temp_vertex_list[3] = rectList[i].estimate_point(rectList[j].points[3, :2])

                        thisRect = asPolygon(temp_vertex_list)

                        stop_adding=False

                        for other_rect in local_rect_list:
                            if thisRect.within(other_rect):
                                stop_adding=True

                        if not stop_adding:
                            local_rect_list.append(thisRect)

                            localVertexList = np.append( localVertexList,temp_vertex_list,axis=0)

                            temp_segment_list = [[segLen,segLen+1],
                                        [segLen + 1, segLen + 2],
                                        [segLen + 2, segLen + 3],
                                        [segLen + 3, segLen]]

                            localSegmentList = np.append(localSegmentList,temp_segment_list,axis=0)


                triangulation_input = {}
                localSegmentList = np.delete(localSegmentList,0,0)

                triangulation_input['vertices'] = localVertexList[:, 0:2]
                if len(localSegmentList) > 0 :
                    triangulation_input['segments'] = localSegmentList

                tria = tri.triangulate(triangulation_input)

                triangle_offset = len(globalVertexList)

                for triangle in tria['triangles']:

                    vertices = np.array([localVertexList[triangle[0]], localVertexList[triangle[1]], localVertexList[triangle[2]]])
                    trianglePolygon = asPolygon(vertices)

                    add_triangle = True

                    for rect in local_rect_list:
                        if trianglePolygon.within(rect):
                            add_triangle = False

                    if add_triangle:
                        triangle_to_add = triangle + triangle_offset
                        faceList.append(triangle_to_add)

                globalVertexList = np.append(globalVertexList, localVertexList, axis=0)

            globalVertexList = np.delete(globalVertexList, 0, 0)
            textureCoordinates = np.copy(globalVertexList[:,:2])

            textureCoordinates[:, 0] = -1*(textureCoordinates[:, 0]/float(intensity_img.shape[0]))+1
            textureCoordinates[:, 1] = textureCoordinates[:, 1]/float(intensity_img.shape[1])

            make_obj_file("model.obj", globalVertexList, faceList,textureCoordinates,"material.mtl","Kaula")
            print("model written to file")


        pressedKey = cv2.waitKey(10)

    cv2.destroyAllWindows()

if __name__== "__main__":
  main()