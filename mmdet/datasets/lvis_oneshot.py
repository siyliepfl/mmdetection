# Copyright (c) OpenMMLab. All rights reserved.

from .coco_oneshot import CocoOneShotDataset

import itertools
import logging
import os.path as osp
import torch
import tempfile
import warnings
from collections import OrderedDict
from mmcv.utils import print_log
from terminaltables import AsciiTable
from .builder import DATASETS
from .pipelines import Compose
import random

import copy
from PIL import ImageFilter, Image
from mmdet.models.utils.box_ops import box_xywh_to_xyxy
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from mmcv.image.photometric import *
from mmdet.core.visualization import imshow_det_bboxes
import matplotlib.pyplot as plt

def debug_plot(data):
    img_ori = imdenormalize(np.array(data['img'].data.permute(1, 2, 0)),
                            mean=np.array([123.675, 116.28, 103.53]),
                            std=np.array([58.395, 57.12, 57.375]))
    crop_im = imshow_det_bboxes(img_ori,
                                np.array(data['query_targets'].data).reshape(-1,4),
                                np.array([0]),
                                segms=None)

    qimg_ori = imdenormalize(np.array(data['query_img'].data.permute(1, 2, 0)),
                             mean=np.array([123.675, 116.28, 103.53]),
                             std=np.array([58.395, 57.12, 57.375]))

    plt.imshow(qimg_ori/255)
    plt.show()

def get_random_patch_from_img(im_size, gt_bboxes=None, min_pixel=8, bg_gt_overlap_iou=0.3):

    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    while True:
        h, w = im_size
        min_w, max_w = min_pixel, w - min_pixel
        min_h, max_h = min_pixel, h - min_pixel
        sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
        x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0

        bbox = torch.tensor([x, y, sw, sh])
        if gt_bboxes is not None:
            ious = bbox_overlaps(np.array(gt_bboxes), np.array(box_xywh_to_xyxy(bbox)).reshape(-1, 4))
            if (ious > bg_gt_overlap_iou).any():
                continue
            else:
                return x, y, sw, sh

        return  x, y, sw, sh

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


@DATASETS.register_module()
class LVISV1OneShotDataset(CocoOneShotDataset):

    CLASSES = (
        'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol',
        'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna',
        'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
        'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
        'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
        'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
        'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
        'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
        'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
        'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
        'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
        'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
        'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
        'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
        'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
        'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
        'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
        'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
        'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
        'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
        'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
        'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
        'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
        'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
        'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)',
        'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box',
        'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere',
        'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase',
        'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts',
        'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer',
        'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn',
        'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card',
        'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car',
        'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf',
        'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)',
        'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar',
        'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup',
        'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
        'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
        'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
        'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
        'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower',
        'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone',
        'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier',
        'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard',
        'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime',
        'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar',
        'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker',
        'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider',
        'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet',
        'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine',
        'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock',
        'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster',
        'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach',
        'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table',
        'coffeepot', 'coil', 'coin', 'colander', 'coleslaw',
        'coloring_material', 'combination_lock', 'pacifier', 'comic_book',
        'compass', 'computer_keyboard', 'condiment', 'cone', 'control',
        'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
        'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
        'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
        'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
        'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
        'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
        'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
        'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
        'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
        'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
        'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
        'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
        'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux',
        'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
        'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
        'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
        'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly',
        'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit',
        'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)',
        'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell',
        'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring',
        'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater',
        'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk',
        'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan',
        'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)',
        'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm',
        'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace',
        'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl',
        'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap',
        'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)',
        'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
        'folding_chair', 'food_processor', 'football_(American)',
        'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car',
        'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice',
        'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage',
        'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic',
        'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator',
        'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture',
        'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
        'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose',
        'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat',
        'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly',
        'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet',
        'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock',
        'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel',
        'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw',
        'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband',
        'headboard', 'headlight', 'headscarf', 'headset',
        'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
        'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
        'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
        'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
        'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
        'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
        'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
        'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
        'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
        'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
        'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
        'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
        'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
        'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
        'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
        'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce',
        'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
        'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
        'lizard', 'log', 'lollipop', 'speaker_(stereo_equipment)', 'loveseat',
        'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
        'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger',
        'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato',
        'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox',
        'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine',
        'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone',
        'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror',
        'mitten', 'mixer_(kitchen_tool)', 'money',
        'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
        'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)',
        'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
        'music_stool', 'musical_instrument', 'nailfile', 'napkin',
        'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper',
        'newsstand', 'nightshirt', 'nosebag_(for_animals)',
        'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker',
        'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
        'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich',
        'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad',
        'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas',
        'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake',
        'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
        'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol',
        'parchment', 'parka', 'parking_meter', 'parrot',
        'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
        'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
        'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
        'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
        'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
        'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
        'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
        'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
        'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
        'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
        'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
        'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
        'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
        'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
        'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato',
        'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel',
        'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune',
        'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher',
        'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit',
        'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish',
        'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat',
        'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
        'recliner', 'record_player', 'reflector', 'remote_control',
        'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
        'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
        'rolling_pin', 'root_beer', 'router_(computer_equipment)',
        'rubber_band', 'runner_(carpet)', 'plastic_bag',
        'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
        'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
        'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
        'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
        'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
        'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
        'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
        'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
        'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl',
        'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
        'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
        'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
        'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
        'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
        'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
        'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
        'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
        'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
        'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
        'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
        'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
        'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
        'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
        'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer',
        'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer',
        'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign',
        'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl',
        'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses',
        'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband',
        'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword',
        'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
        'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
        'tambourine', 'army_tank', 'tank_(storage_vessel)',
        'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
        'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
        'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
        'telephone_pole', 'telephoto_lens', 'television_camera',
        'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
        'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
        'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil',
        'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven',
        'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush',
        'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel',
        'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light',
        'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline',
        'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle',
        'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat',
        'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
        'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
        'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
        'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
        'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
        'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
        'washbasin', 'automatic_washer', 'watch', 'water_bottle',
        'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
        'water_gun', 'water_scooter', 'water_ski', 'water_tower',
        'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
        'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
        'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
        'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
        'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
        'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
        'yoke_(animal_equipment)', 'zebra', 'zucchini')


    def __init__(self, query_pipeline,
                 split,
                 average_num,
                 query_json=None,
                 no_test_class_present=False,
                 bg_crop_freq=0.25,
                 bg_gt_overlap_iou=0.3,
                 **kwargs):
        super(CocoOneShotDataset, self).__init__( **kwargs)

        if self.test_mode and query_json is not None:
            self.class_anno_mapping, self.imid_info_mapping_dict = self.build_query_bank_from_files(query_json)
        else:
            self.class_anno_mapping = self.build_class_anno_mapping()

        self.query_json = query_json
        self.label2cat = {y:x for x,y in self.cat2label.items()}

        if self.test_mode:
            self.pipeline = Compose(kwargs['pipeline'])
        else:
            self.load_pipeline = Compose(kwargs['pipeline'][:2])
            self.transform_pipeline = Compose(kwargs['pipeline'][2:-1])
            bg_transform_pipeline = copy.deepcopy(kwargs['pipeline'][2:-1])
            bg_transform_pipeline[1] = dict(type='RandomFlip', flip_ratio=0.0)
            self.bg_transform_pipeline = Compose(bg_transform_pipeline)
            self.collect_pipeline = Compose([kwargs['pipeline'][-1]])

        self.bg_query_pipeline = self.get_query_transforms()

        self.query_pipeline = Compose(query_pipeline)
        self.avg_num = average_num
        self.split = split
        self.no_test_class_present = no_test_class_present
        self.bg_crop_freq = bg_crop_freq
        self.bg_gt_overlap_iou = bg_gt_overlap_iou

        self.known_cats_labels = [cat
                                  for cat in range(0, 80)
                                  if cat % 4 != split
                                  ]
        self.unknown_cats_labels = [cat
                                    for cat in range(0, 80)
                                    if cat % 4 == split
                                    ]
        self.known_cats_ids = {
                        self.label2cat[cat_label]
                                 for cat_label in self.known_cats_labels
        }
        self.unknown_cats_ids = {
                        self.label2cat[cat_label]
                                 for cat_label in self.unknown_cats_labels
        }

    def load_annotations(self, ann_file):
        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        self.coco = LVIS(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            # coco_url is used in LVISv1 instead of file_name
            # e.g. http://images.cocodataset.org/train2017/000000391895.jpg
            # train/val split in specified in url
            info['filename'] = info['coco_url'].replace(
                'http://images.cocodataset.org/', '')
            data_infos.append(info)
        return data_infos

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in LVIS protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None):
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: LVIS style metrics.
        """

        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from lvis import LVISEval, LVISResults
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)

        eval_results = OrderedDict()
        # get original api
        lvis_gt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                lvis_dt = LVISResults(lvis_gt, result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type)
            lvis_eval.params.imgIds = self.img_ids
            if metric == 'proposal':
                lvis_eval.params.useCats = 0
                lvis_eval.params.maxDets = list(proposal_nums)
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                for k, v in lvis_eval.get_results().items():
                    if k.startswith('AR'):
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[k] = val
            else:
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                lvis_results = lvis_eval.get_results()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = lvis_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        # the dimensions of precisions are
                        # [num_thrs, num_recalls, num_cats, num_area_rngs]
                        nm = self.coco.load_cats([catId])[0]
                        precision = precisions[:, :, idx, 0]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                for k, v in lvis_results.items():
                    if k.startswith('AP'):
                        key = '{}_{}'.format(metric, k)
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[key] = val
                ap_summary = ' '.join([
                    '{}:{:.3f}'.format(k, float(v))
                    for k, v in lvis_results.items() if k.startswith('AP')
                ])
                eval_results['{}_mAP_copypaste'.format(metric)] = ap_summary
            lvis_eval.print_results()
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

