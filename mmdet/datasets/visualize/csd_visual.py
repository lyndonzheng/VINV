from mmdet.datasets.visualize import CSD
from PIL import Image
import numpy as np

dataDir = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/suncg_data/sosc_new'
dataType = 'train'
annFile = '{}/sosc_new_{}_order.json'.format('/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/suncg_data', dataType)
# dataDir = '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/coco'
# dataType = 'train2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# initializa sosc api for instance annotations
csd = CSD(annFile)
# display coco categories and supercategories
cats = csd.loadCats(csd.getCatIds())
nms=[cat['name'] for cat in cats]
print('SOSC categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('SOSC supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
# imgIds = csd.getImgIds(imgIds=[], catIds=csd.getCatIds(supNms=['soft']))
imgIds = csd.getImgIds(imgIds=[], catIds=csd.getCatIds(supNms=['refridgerator', 'table','door','counter', 'chair']))
# imgIds = csd.getImgIds(imgIds=[41])
# image = csd.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
for i in range(0, len(imgIds)):
    image = csd.loadImgs(imgIds[i])[0]

    # load and display image
    if image['img_name'] == '3b87fe9feb2686a43b770bae92f53c7a/rgb/03.png':
        rgba = Image.open('%s/%s'%(dataDir, image['img_name']))
        rgba.show()
        d = Image.open('%s/%s'%(dataDir, image['depth_name']))
        # d.show()

        # load and display instance annotations
        palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # annIds = sosc.getAnnIds(imgIds=image['id'], catIds=sosc.getCatIds(supNms=['television']))
        annIds = csd.getAnnIds(imgIds=image['id'])
        print(image['id'])
        anns = csd.loadAnns(annIds)
        img = csd.showAnns(rgba, anns, f_bbox=True, v_bbox=False, f_mask=False, name=False, layer=False, v_mask=True, object_id=False, pair_order=False, dataDir=dataDir)
        room_name, _, image_name = image['img_name'].split('/')
        img.save('/media/lyndon/2e91762c-97d9-40c9-9af1-6f318aca47711/results/CVPR20/visualize/'+room_name+'_'+image_name)