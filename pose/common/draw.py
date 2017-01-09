"""
Code to draw images with pose locations.
"""
from PIL import Image, ImageDraw
from tensorflow.python.platform import gfile
import os.path as pt

from pose.common.common_globals import FLAGS
from pose.common.common_globals import BodyParts as p


labels_ = 0
drawer_ = 0


def g(bodyPartId, r=2):
    global labels_
    return (labels_[bodyPartId,0]-r, labels_[bodyPartId,1]-r,
        labels_[bodyPartId,0]+r, labels_[bodyPartId,1]+r)


def draw(startPart,stopPart,fill="red"):
    drawer_.line([g(startPart), g(stopPart)], width=3, fill=fill)


def drawPoseOnImage(fim, argLabels, folder, fileid, fname=None, reshaped=False):
    global labels_
    global drawer_

    if not reshaped:
        labels_ = argLabels.reshape([FLAGS.label_count, FLAGS.label_size])[:,:2]
    else:
        labels_ = argLabels

    im = Image.fromarray(fim)
    drawer_ = ImageDraw.Draw(im)

    drawer_.ellipse(g(p.Head_top), fill='green')
    drawer_.ellipse(g(p.Neck), fill='green')
    drawer_.ellipse(g(p.Left_hip), fill='green')
    drawer_.ellipse(g(p.Right_hip), fill='green')
    drawer_.ellipse(g(p.Left_shoulder), fill='green')
    drawer_.ellipse(g(p.Right_shoulder), fill='green')
    drawer_.ellipse(g(p.Left_elbow), fill='green')
    drawer_.ellipse(g(p.Right_elbow), fill='green')
    drawer_.ellipse(g(p.Left_wrist), fill='green')
    drawer_.ellipse(g(p.Right_wrist), fill='green')
    drawer_.ellipse(g(p.Left_ankle), fill='green')
    drawer_.ellipse(g(p.Right_ankle), fill='green')
    drawer_.ellipse(g(p.Left_knee), fill='green')
    drawer_.ellipse(g(p.Right_knee), fill='green')

    # #middle parts
    # draw(p.Head_top, p.Neck)
    # draw(p.Left_hip, p.Right_hip)

    # #left arm
    # draw(p.Neck, p.Left_shoulder)
    # draw(p.Left_shoulder, p.Left_elbow)
    # draw(p.Left_elbow, p.Left_wrist)

    # #left leg
    # draw(p.Left_shoulder, p.Left_hip)
    # draw(p.Left_hip, p.Left_knee)
    # draw(p.Left_knee, p.Left_ankle)

    # #right arm
    # draw(p.Neck, p.Right_shoulder)
    # draw(p.Right_shoulder, p.Right_elbow)
    # draw(p.Right_elbow, p.Right_wrist)

    # #right leg
    # draw(p.Right_shoulder, p.Right_hip)
    # draw(p.Right_hip, p.Right_knee)
    # draw(p.Right_knee, p.Right_ankle)


    if not gfile.Exists(folder):
        gfile.MakeDirs(folder)

    if fname is not None:
        im.save(pt.join(folder, fname))
    else:
        im.save(pt.join(folder, "%05d.jpg" % fileid))

    return


def showPoseOnImage(im, argLabels):
    global labels_
    global drawer_

    labels_ = argLabels
    drawer_ = ImageDraw.Draw(im)

    drawer_.ellipse(g(p.Head_top), fill='green')
    drawer_.ellipse(g(p.Neck), fill='green')
    drawer_.ellipse(g(p.Left_hip), fill='green')
    drawer_.ellipse(g(p.Right_hip), fill='green')
    drawer_.ellipse(g(p.Left_shoulder), fill='green')
    drawer_.ellipse(g(p.Right_shoulder), fill='green')
    drawer_.ellipse(g(p.Left_elbow), fill='green')
    drawer_.ellipse(g(p.Right_elbow), fill='green')
    drawer_.ellipse(g(p.Left_wrist), fill='green')
    drawer_.ellipse(g(p.Right_wrist), fill='green')
    drawer_.ellipse(g(p.Left_ankle), fill='green')
    drawer_.ellipse(g(p.Right_ankle), fill='green')
    drawer_.ellipse(g(p.Left_knee), fill='green')
    drawer_.ellipse(g(p.Right_knee), fill='green')

    im.show()
