#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" CNN modules for face verification """

# training set triplet generator
def train_triplet_generator(df, batch_size=100, img_size=(224, 224), seed=42, 
                            prefix='./data/triplet/train/'):
    """ training set triplet generator
        it will generate 7400 triplet images in total
    """
    # get images with only one training image landmark id and the rest landmark ids
    np.random.seed(seed)
    grouped = df[['landmark_id', 'image_id']].groupby('landmark_id').count().reset_index()
    unique_neg_ids = list(grouped[grouped['image_id'] == 1]['landmark_id'].values)
    rest_ids = list(grouped[grouped['image_id'] > 1]['landmark_id'].values)
    size = 7400 * 2 - len(unique_neg_ids) 
    zeros = np.zeros((batch_size, 3, 1), dtype=K.floatx())
    
    while True:
        # get positive and negative image landmark ids
        np.random.shuffle(rest_ids)
        candidate_ids = list(np.random.choice(rest_ids, size=size, replace=False))
        pos_landmark_ids = candidate_ids[:7400]
        neg_landmark_ids = candidate_ids[7400:] + unique_neg_ids
        np.random.shuffle(neg_landmark_ids)
        
        # transform landmark id into image id
        anc_img_ids = []
        pos_img_ids = []
        neg_img_ids = []
        
        for i in range(len(pos_landmark_ids)):
            tmp_pos_ids = df[df['landmark_id'] == pos_landmark_ids[i]]['image_id'].values
            anc_img_ids.append(tmp_pos_ids[0])
            pos_img_ids.append(tmp_pos_ids[1])
            
            tmp_neg_ids = df[df['landmark_id'] == neg_landmark_ids[i]]['image_id'].values
            neg_img_ids.append(tmp_neg_ids[0])
        
        # iterator to read batch images
        for j in range(len(pos_img_ids) // batch_size):
            batch_anc_img_ids = anc_img_ids[j * batch_size: (j + 1) * batch_size]
            batch_pos_img_ids = pos_img_ids[j * batch_size: (j + 1) * batch_size]
            batch_neg_img_ids = neg_img_ids[j * batch_size: (j + 1) * batch_size]
            
            # get images
            anc_imgs = []
            pos_imgs = []
            neg_imgs = []
            
            # iteratively read images
            for k in range(batch_size):
                anc_path = prefix + str(batch_anc_img_ids[k]) + '.jpg'
                pos_path = prefix + str(batch_pos_img_ids[k]) + '.jpg'
                neg_path = prefix + str(batch_neg_img_ids[k]) + '.jpg'
                
                tmp_anc_img = load_img(anc_path, target_size=img_size)
                tmp_anc_img = img_to_array(tmp_anc_img)
                anc_imgs.append(tmp_anc_img)
                
                tmp_pos_img = load_img(pos_path, target_size=img_size)
                tmp_pos_img = img_to_array(tmp_pos_img)
                pos_imgs.append(tmp_pos_img)
                
                tmp_neg_img = load_img(neg_path, target_size=img_size)
                tmp_neg_img = img_to_array(tmp_neg_img)
                neg_imgs.append(tmp_neg_img)
        
            # transform list to array
            anc_imgs = np.array(anc_imgs, dtype=K.floatx()) / 255.0
            pos_imgs = np.array(pos_imgs, dtype=K.floatx()) / 255.0
            neg_imgs = np.array(neg_imgs, dtype=K.floatx()) / 255.0

            yield [anc_imgs, pos_imgs, neg_imgs], zeros
            

# validation set triplet generator
def val_triplet_generator(df, batch_size=128, img_size=(224, 224), 
                          seed=42, prefix='./data/triplet/validation'):
    """ validation set triplet collector """
    
     # get images with only one image landmark id and the rest landmark ids
    grouped = df[['landmark_id', 'image_id']].groupby('landmark_id').count().reset_index()
    unique_neg_ids = list(grouped[grouped['image_id'] == 1]['landmark_id'].values)
    rest_ids = list(grouped[grouped['image_id'] > 1]['landmark_id'].values)
    size = 3072 * 2 - len(unique_neg_ids) 
    zeros = np.zeros((batch_size, 3, 1), dtype=K.floatx())
    
    while True:
        # get positive and negative image landmark ids
        np.random.seed(seed)
        candidate_ids = list(np.random.choice(rest_ids, size=size, replace=False))
        pos_landmark_ids = candidate_ids[:3072]
        neg_landmark_ids = candidate_ids[3072:] + unique_neg_ids
        np.random.shuffle(neg_landmark_ids)
        
        # transform landmark id into image id
        anc_img_ids = []
        pos_img_ids = []
        neg_img_ids = []
        
        for i in range(len(pos_landmark_ids)):
            tmp_pos_ids = df[df['landmark_id'] == pos_landmark_ids[i]]['image_id'].values
            anc_img_ids.append(tmp_pos_ids[0])
            pos_img_ids.append(tmp_pos_ids[1])
            
            tmp_neg_ids = df[df['landmark_id'] == neg_landmark_ids[i]]['image_id'].values
            neg_img_ids.append(tmp_neg_ids[0])
        
        # iterator to read batch images
        for j in range(len(pos_img_ids) // batch_size):
            batch_anc_img_ids = anc_img_ids[j * batch_size: (j + 1) * batch_size]
            batch_pos_img_ids = pos_img_ids[j * batch_size: (j + 1) * batch_size]
            batch_neg_img_ids = neg_img_ids[j * batch_size: (j + 1) * batch_size]
            
            # get images
            anc_imgs = []
            pos_imgs = []
            neg_imgs = []
            
            # iteratively read images
            for k in range(batch_size):
                anc_path = prefix + str(batch_anc_img_ids[k]) + '.jpg'
                pos_path = prefix + str(batch_pos_img_ids[k]) + '.jpg'
                neg_path = prefix + str(batch_neg_img_ids[k]) + '.jpg'
                
                tmp_anc_img = load_img(anc_path, target_size=img_size)
                tmp_anc_img = img_to_array(tmp_anc_img)
                anc_imgs.append(tmp_anc_img)
                
                tmp_pos_img = load_img(pos_path, target_size=img_size)
                tmp_pos_img = img_to_array(tmp_pos_img)
                pos_imgs.append(tmp_pos_img)
                
                tmp_neg_img = load_img(neg_path, target_size=img_size)
                tmp_neg_img = img_to_array(tmp_neg_img)
                neg_imgs.append(tmp_neg_img)
        
            # transform list to array
            anc_imgs = np.array(anc_imgs, dtype=K.floatx()) / 255.0
            pos_imgs = np.array(pos_imgs, dtype=K.floatx()) / 255.0
            neg_imgs = np.array(neg_imgs, dtype=K.floatx()) / 255.0
            
            yield [anc_imgs, pos_imgs, neg_imgs], zeros