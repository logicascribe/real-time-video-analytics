#!/usr/bin/python3
import cv2
import time
import argparse
import math
from pathlib import Path
from ultralytics import YOLO
from tracker import IoUTracker, TrackingCriteria, DEFAULT_TRACKING_CRITERIA, TrackingStatus, TrackStats, Track


###########################
## Constants definitions ##
###########################
NORMAL_DRAW = 1
BOLT_DRAW   = 2
GREEN_COLOR    = (0, 255, 0)
BLUE_COLOR     = (255, 0, 0)
RED_COLOR      = (0, 0, 255)
YELLOW_COLOR   = (0, 255, 255)
PINK_COLOR     = (255, 0, 255)
WHITE_COLOR    = (255, 255, 255)
ORANGE_COLOR   = (0, 128, 255)
#TRACKING_WINDOW_DEFAULT = 3


###########################
## Functions definitions ##
###########################

def extract_detections(result, allowed_classes, confidence_threshold):
    detections = []
    for box, confidence, class_id in zip(result.boxes.xyxy,
                                         result.boxes.conf,
                                         result.boxes.cls):
        class_id = int(class_id)
        class_name = result.names[class_id]

        if class_name in allowed_classes and (confidence >= confidence_threshold) :
            detections.append([class_name, float(confidence), box.tolist()])
    return detections

def draw_detection(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[2])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            BLUE_COLOR,
            NORMAL_DRAW
        )

        text = f"{det[0]} {det[1]:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )


def draw_tracks(frame, tracks):
    statusColorCode = {
        TrackingStatus.ToDelete : RED_COLOR,
        TrackingStatus.Tentative : ORANGE_COLOR,
        TrackingStatus.Tracked : GREEN_COLOR
        }
    for track in tracks:
        trackColorCode = statusColorCode[track.status]
        x1, y1, x2, y2 = map(int, track.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            trackColorCode,
            NORMAL_DRAW
        )

        text = f"Id:{track.id} Class:{track.class_name} Conf.:{track.confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            trackColorCode,
            1
        )


def testSingleImage(frame, allowed_classes):
    results = model(imagePath)
    result = results[0]
    detections = extract_detections(
        result,
        allowed_classes,
        0.5
        )
    draw_detection(image, detections)
    cv2.imshow("Manipulation", frame)

def testTracker():
#    frame0 = [
#        ["car", 0.9, [10, 10, 30, 30]]
#        ]
#    frame1 = [
#        ["car", 0.9, [12, 10, 32, 30]]
#        ]
#    frame2 = [
#        ["car", 0.9, [14, 10, 34, 30]]
#        ]

    frames_2cars = [
        [
        ["car", 0.9, [10, 10, 30, 30]]
        ],
        [
        ["car", 0.9, [12, 10, 32, 30]]
        ],
        [
        ],
        [
        ],
        [
        ["car", 0.9, [24, 10, 54, 30]]
        ]  ,
        [
        ["car", 0.9, [84, 10, 124, 30]]
        ],
        [
        ["car", 0.9, [24, 10, 54, 30]],
        ["car", 0.9, [84, 10, 124, 30]]
        ]

    ]
    #frames = [frame0, frame1, frame2]
    myIoUTracker = IoUTracker()

#    for frame in frames:
    for frame in frames_2cars:
        myIoUTracker.update(frame)
        myIoUTracker.displayTracks()
        print()


def test_detection(source_video_file, tracking_criteria, display_video = False, display_time_stat = False, display_track_stat = False, frameIncr = 1, timeSleep = 0):
    waitBeforeNextFrame = (0 != timeSleep)
    frameCount = 0
    nextFrame = 0
    if frameIncr < 1:
        raise ValueError("frameIncr must be >= 1")

    pcap = cv2.VideoCapture(source_video_file)
    # Vérifier si le flux s'est ouvert correctement
    if not pcap.isOpened():
        raise IOError(f"Impossible d'ouvrir le flux vidéo ou la caméra : {source_video_file}")
    largeur = int(pcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(pcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Résolution de la capture : {largeur}x{hauteur}")

    model = YOLO("yolo26n.pt")
    allowed_classes = ["person",
                       "bicycle",
                       "car",
                       "motorcycle",
                       "bus",
                       "truck"]

    sum_inference_time = 0
    sum_processing_time = 0
    processed_frames = 0
    sum_end_to_end_time = 0
    tracker = IoUTracker(tracking_criteria)

    while True:
        t_start = time.perf_counter()
        ret, frame = pcap.read()
        if not ret:
            current = int(pcap.get(cv2.CAP_PROP_POS_FRAMES))
            total = int(pcap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"End of video: position={current}, total={total}")
            break

        if (frameCount != nextFrame):
            frameCount += 1
            continue
        else:
            nextFrame += frameIncr
            frameCount += 1

        t0 = time.perf_counter()
        results = model(frame, verbose=False)
        t1 = time.perf_counter()

        result = results[0]

        detections = extract_detections(result, allowed_classes, 0.1)
        tracker.update(detections)

        draw_tracks(frame, tracker.tracks)

        if display_video:
            cv2.imshow("Video Analytics", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        t2 = time.perf_counter()
        t_inference = t1 - t0
        t_processing = t2 - t0
        t_end_to_end = t2 - t_start
        sum_inference_time += t_inference
        sum_processing_time += t_processing
        sum_end_to_end_time += t_end_to_end
        processed_frames += 1
        if waitBeforeNextFrame:
            time.sleep(timeSleep)

    processing_fps = math.nan if (0 == sum_processing_time) else processed_frames / sum_processing_time
    mean_processing_latency = math.nan if (0 == processed_frames) else sum_processing_time / processed_frames
    mean_inference_latency = math.nan if (0 == processed_frames) else sum_inference_time / processed_frames
    mean_sum_end_to_end_time = math.nan if (0 == processed_frames) else sum_end_to_end_time / processed_frames
    if (display_time_stat):
        print ("mean end to end processing time = ", mean_sum_end_to_end_time)
        print ("number of processed frames = ", processed_frames)
        print ("elapsed processing time = ", sum_processing_time)
        print ("mean processing latency = ", mean_processing_latency)
        print ("mean processing FPS = ", processing_fps)
        print ("mean inference latency = ", mean_inference_latency)

    cv2.destroyAllWindows()

    track_stat = tracker.update_get_tracks_stat()
    print ()
    tracker.qa_tracks_analyze.displayTracksStats()



###########################
## Starting main program ##
###########################
DEFAULT_VIDEO_SOURCE = "../resources/video_trafic_voitures.mp4"
NO_TEST = "None"
TEST_TRACKER = "testTracker"
video_source = DEFAULT_VIDEO_SOURCE
imagePath = "./traffic_frame.jpg"
parser = argparse.ArgumentParser(description="motion_detection")

UsualTrackingCriteria = TrackingCriteria(
    iou_threshold=0.25,
    tracking_windows=10,
    min_hits_frames=7,   # ceil(2/3 * 10)
    max_missed_frames=10
)

# TODO: Use Command pattern recorded in a dictionnary to efficiently store, display and execute all the tests, especially in help messages.
# Argument optionnel avec valeur par défaut
parser.add_argument("-t", "--test", type = str, default = NO_TEST, help="Identifiant du test à effectuer (0: Aucun test)")
parser.add_argument("-v", "--video_source", type = str, default = DEFAULT_VIDEO_SOURCE, help="File to process")
parser.add_argument("-f", "--frame_incrementation", type = int, default = 1, help="Frame incrementation")
parser.add_argument("-d", "--delay", type = int, default = 0, help="Delay between 2 consecutive frames incrementation (in second)")
parser.add_argument("-n", "--no_display", action="store_true", help="Show the frame or not")
parser.add_argument("--loop", action="store_true", help="Show the frame or not")

args = parser.parse_args()

performTest = (NO_TEST != args.test)
video_source = args.video_source

frame_incrementation = args.frame_incrementation
timeSleepArg = args.delay
display_frame = not args.no_display
loop = args.loop

if performTest:
    if (TEST_TRACKER == args.test):
        testTracker()
    quit()

print(f"video_source = {video_source}")

if loop:
    windows = [3, 5, 7, 10, 15]
    iou_thresholds = [0.15, 0.20, 0.25, 0.30, 0.40]
    MIN_HIT_RATIO = 2 / 3
    for win in windows:
        for it_iou_threshold in iou_thresholds:
            min_hits = math.ceil(win * MIN_HIT_RATIO)
            max_missed_frames = win
            tracking_criteria = TrackingCriteria(iou_threshold = it_iou_threshold, tracking_windows = win, min_hits_frames = min_hits, max_missed_frames = max_missed_frames)
            label = tracking_criteria.label()
            print(f"label = {label}")
            print(f"iou_threshold = {it_iou_threshold}")
            print(f"tracking_window = {win}")
            print(f"minimum_hit = {min_hits}")
            print(f"max_missed_frames = {max_missed_frames}")
            print("=>")
            test_detection(video_source, tracking_criteria, timeSleep = timeSleepArg, frameIncr = frame_incrementation, display_video = display_frame, display_time_stat = False, display_track_stat = True)
            print("")
            print("")
else:
    test_detection(video_source, UsualTrackingCriteria, timeSleep = timeSleepArg, frameIncr = frame_incrementation, display_video = display_frame, display_time_stat = False, display_track_stat = True)
