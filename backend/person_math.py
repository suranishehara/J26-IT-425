"""Image-space estimates and colour sampling; not physical measurements."""
import numpy as np

# Values verified against Google's published colour-block component.
MST_HEX=['#f6ede4','#f3e7db','#f7ead0','#eadaba','#d7bd96','#a07e56','#825c43','#604134','#3a312a','#292420']
MST_SOURCE='https://skintone.google/assets/tono-color-block-arranger.73c427e1.js'

def rgb_to_lab(rgb):
    rgb=np.asarray(rgb,dtype=float)/255
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    xyz=linear@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
    xyz=xyz/np.array([.95047,1,1.08883])
    delta=6/29
    f=np.where(xyz>delta**3,np.cbrt(xyz),xyz/(3*delta**2)+4/29)
    return np.stack((116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])),axis=-1)

def estimate_skin(rgb,labels,confidence):
    from scipy import ndimage
    mask=(labels==11)&(confidence>=.65)
    components,count=ndimage.label(mask)
    if not count:
        return {'status':'unavailable','reason':'No clear face region detected.'},np.zeros_like(mask)
    sizes=np.bincount(components.ravel());sizes[0]=0
    mask=components==sizes.argmax()
    yy,xx=np.where(mask);height=int(yy.max()-yy.min()+1);width=int(xx.max()-xx.min()+1)
    if height<24 or width<16:
        return {'status':'unavailable','reason':'Face is too small. Try a closer, evenly lit photo.'},np.zeros_like(mask)
    y,x=np.indices(mask.shape);rx=(x-xx.min())/width;ry=(y-yy.min())/height
    cheeks=mask&(ry>=.42)&(ry<=.67)&(((rx>=.15)&(rx<=.36))|((rx>=.64)&(rx<=.85)))
    pixels=rgb[cheeks]
    if len(pixels)<30:
        return {'status':'unavailable','reason':'Not enough clear cheek pixels.'},cheeks
    median=np.median(pixels,axis=0)
    references=np.array([[int(value[i:i+2],16) for i in (1,3,5)] for value in MST_HEX])
    distances=np.linalg.norm(rgb_to_lab(references)-rgb_to_lab(median),axis=1)
    order=distances.argsort()
    lightness=rgb_to_lab(pixels)[:,0]
    spread=float(np.percentile(lightness,90)-np.percentile(lightness,10))
    clipped=float(np.mean(np.all(pixels<=5,axis=1)|np.all(pixels>=250,axis=1)))
    warnings=['Uncalibrated photo appearance; lighting, camera white balance and makeup can change the estimate.']
    if spread>20:
        warnings.append('Uneven illumination or face-region variation detected.')
    if clipped>.05:
        warnings.append('Some sampled pixels are clipped by exposure.')
    return {'status':'estimated','suggested_mst':int(order[0]+1),
        'sample_hex':'#'+''.join(format(int(round(v)),'02x') for v in median),
        'sample_pixels':len(pixels),'lightness_spread':round(spread,2),
        'alternatives':[{'mst':int(i+1),'colour_distance':round(float(distances[i]),2)} for i in order[:3]],
        'uncertainty':'unvalidated','calibrated':False,'needs_review':True,'warnings':warnings,
        'method':'Median cheek RGB; nearest reference in CIELAB using Euclidean distance, not a learned skin-tone classifier.'},cheeks

def pose_proportions(points,width,height):
    if len(points)<25:
        return {'status':'unavailable','reason':'No reliable body landmarks.'}
    ids=(11,12,23,24)
    if any(points[i].get('visibility',0)<.6 or not (0<=points[i]['x']<=1 and 0<=points[i]['y']<=1) for i in ids):
        return {'status':'unavailable','reason':'Shoulders or hips are hidden or outside the photo.'}
    xy=np.array([[points[i]['x']*width,points[i]['y']*height] for i in ids])
    shoulders=float(np.linalg.norm(xy[0]-xy[1]));hips=float(np.linalg.norm(xy[2]-xy[3]))
    torso=float(np.linalg.norm(xy[:2].mean(0)-xy[2:].mean(0)))
    if min(shoulders,hips,torso)<10:
        return {'status':'unavailable','reason':'Person is too small or pose is unsuitable.'}
    if abs(xy[0,1]-xy[1,1])>shoulders*.3:
        return {'status':'unavailable','reason':'Shoulders are tilted. Use a straight front-facing pose.'}
    return {'status':'estimated','projected_shoulder_hip_ratio':round(shoulders/hips,3),
        'landmark_visibility_min':round(min(points[i]['visibility'] for i in ids),3),
        'needs_review':True,'uncertainty':'unvalidated',
        'notice':'2D landmark span ratio only. It is not a body-shape diagnosis, circumference, or tailoring measurement; perspective and clothing affect it.'}
