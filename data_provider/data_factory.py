# from .Loader import Loader
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch
from .batch_scheduler import WeightBatchSchedulerSampler,DistributedWeightBatchSchedulerSampler
from .data_loader import Dataloader, CXR_ECG_MatchedDataset, ECG_Dataset
from .uea import collate_fn

train_data_dict_UEA = [
    'ArticularyWordRecognition',
    'AtrialFibrillation',
    'BasicMotions',
    'CharacterTrajectories',
    'Cricket',
    # 'DuckDuckGeese',
    # 'EigenWorms',
    'Epilepsy',
    'EthanolConcentration',
    'ERing',
    'FaceDetection',
    'FingerMovements',
    'HandMovementDirection',
    'Handwriting',
    'Heartbeat',
    'InsectWingbeat',
    'JapaneseVowels',
    'Libras',
    'LSST',
    'MotorImagery',
    'NATOPS',
    'PenDigits',
    'PEMS-SF',
    'PhonemeSpectra',
    'RacketSports',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'SpokenArabicDigits',
    'StandWalkJump',
    'UWaveGestureLibrary',
]

train_data_dict_UCR = [
    'ACSF1',
    'Adiac',
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'ArrowHead',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'BME',
    'Car',
    'CBF',
    'Chinatown',
    'ChlorineConcentration',
    'CinCECGTorso',
    'Coffee',
    'Computers',
    'CricketX',
    'CricketY',
    'CricketZ',
    'Crop',
    'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW',
    'Earthquakes',
    'ECG200',
    'ECG5000',
    'ECGFiveDays',
    'ElectricDevices',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'EthanolLevel',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    'FiftyWords',
    'Fish',
    'FordA',
    'FordB',
    'FreezerRegularTrain',
    'FreezerSmallTrain',
    'Fungi',
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GunPoint',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'Ham',
    'HandOutlines',
    'Haptics',
    'Herring',
    'HouseTwenty',
    'InlineSkate',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'InsectWingbeatSound',
    'ItalyPowerDemand',
    'LargeKitchenAppliances',
    'Lightning2',
    'Lightning7',
    'Mallat',
    'Meat',
    'MedicalImages',
    'MelbournePedestrian',
    'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW',
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'MoteStrain',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    'OliveOil',
    'OSULeaf',
    'PhalangesOutlinesCorrect',
    'Phoneme',
    'PickupGestureWiimoteZ',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'PLAID',
    'Plane',
    'PowerCons',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect',
    'ProximalPhalanxTW',
    'RefrigerationDevices',
    'Rock',
    'ScreenType',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'ShapeletSim',
    'ShapesAll',
    'SmallKitchenAppliances',
    'SmoothSubspace',
    'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2',
    'StarLightCurves',
    'Strawberry',
    'SwedishLeaf',
    'Symbols',
    'SyntheticControl',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    'TwoLeadECG',
    'TwoPatterns',
    'UMD',
    'UWaveGestureLibraryAll',
    'UWaveGestureLibraryX',
    'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ',
    'Wafer',
    'Wine',
    'WordSynonyms',
    'Worms',
    'WormsTwoClass',
    'Yoga',
    'DodgerLoopDay',
    'DodgerLoopGame',
    'DodgerLoopWeekend',
]

train_data_dict_monash = [
    'AppliancesEnergy',
    'AustraliaRainfall',
    'BeijingPM10Quality',
    'BeijingPM25Quality',
    'BenzeneConcentration',
    'BIDMC32HR',
    'BIDMC32RR',
    'BIDMC32SpO2',
    'Covid3Month',
    'FloodModeling1',
    'FloodModeling2',
    'FloodModeling3',
    'HouseholdPowerConsumption1',
    'HouseholdPowerConsumption2',
    'IEEEPPG',
    'LiveFuelMoistureContent',
    'NewsHeadlineSentiment',
    'NewsTitleSentiment',
    # 'PPGDalia', # 此数据集过长，容易显存爆炸
]

monash_p5_distance_floor = {
    'AppliancesEnergy': 4,
    'AustraliaRainfall': 1,
    'BIDMC32HR': 17,
    'BIDMC32RR': 17,
    'BIDMC32SpO2': 17,
    'BeijingPM10Quality': 1,
    'BeijingPM25Quality': 1,
    'BenzeneConcentration': 14,
    'Covid3Month': 0,
    'FloodModeling1': 13,
    'FloodModeling2': 11,
    'FloodModeling3': 13,
    'HouseholdPowerConsumption1': 24,
    'HouseholdPowerConsumption2': 24,
    'IEEEPPG': 18,
    'LiveFuelMoistureContent': 4,
    'NewsHeadlineSentiment': 0,
    'NewsTitleSentiment': 0,
}

# before pretrain stage, concat all datasets into one dataset(data mixing)
def train_data_provider(args, flag, hdf5_file_path=None, subnum=None):
    if args.loader in ['CXR_ECG', 'ECG']:
        if args.loader == 'CXR_ECG':
            hdf5_path = hdf5_file_path if hdf5_file_path is not None else args.cxr_ecg_h5
            dataset = CXR_ECG_MatchedDataset(
                cfg=args,
                hdf5_file_path=hdf5_path
            )
            if subnum is not None:
                dataset = Subset(dataset, range(subnum))
        else: # ECG
            csv_path = getattr(args, 'ecg_csv', '/media/omnisky/Disk8.0T/rj/data/MIMIC/processed_csv/df_ecg_pretrain_train.csv')
            ecg_root = getattr(args, 'ecg_root_path', '/media/omnisky/Disk8.0T/rj/data/MIMIC/mimic-iv-ecg/1.0/files')
            dataset = ECG_Dataset(
                csv_path=csv_path,
                ecg_root_path=ecg_root
            )
        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f'{args.loader} dataset ({flag}) size: ', len(dataset))
        
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if getattr(args, "distributed", False) or torch.distributed.is_initialized() else None
        
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=(flag == 'TRAIN'),
            pin_memory=True,
            persistent_workers=True
        )
        return dataset, data_loader

    # choose which dataset to use
    if args.loader == 'UEA':
        train_data_dict = train_data_dict_UEA
    elif args.loader == 'UCR':
        train_data_dict = train_data_dict_UCR
    else:
        train_data_dict = train_data_dict_monash

    data_sum = len(train_data_dict)
    concat_dataset = []
    weights = [] # dataset's sample ratio is determined by weight
    train_size = 0
    batch_size = args.batch_size
    num_workers = args.num_workers
    drop_last = False

    i = 0
    for dataset_name in train_data_dict:
        i += 1
        dataset = Dataloader(
            loader=args.loader,
            dataset_name=dataset_name,
            flag=flag,
            data_path=args.data_path
        )
        concat_dataset.append(dataset)
        weights.append(len(dataset))
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f'{dataset_name}  size: ', len(dataset))
        train_size += len(dataset)
        if i == data_sum:    break
        
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('Pretrain data number: {0} | Pretrain data size: {1}'.format(data_sum, train_size))
    
    concat_dataset = ConcatDataset(concat_dataset)
    weights = [i / train_size for i in weights]

    if args.sampler_type == 'weighted':
        sampler=WeightBatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size, train_size=train_size, weights=weights)
    else:
        sampler=DistributedWeightBatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size, train_size=train_size, weights=weights)
        
    data_loader = DataLoader(
        dataset=concat_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x),
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True
    )
    
    return concat_dataset, data_loader