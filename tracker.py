import cv2
import math
import time
import argparse
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any
import more_itertools
from more_itertools import partition
from IoUTools.compute_iou import compute_iou, compute_iou_matrix, optimal_match_from_score_matrix
from IoUTools.compute_iou import IDX_DET_CLASS, IDX_DET_CONFIDENCE, IDX_DET_BBOX
from statistics import median
from collections import Counter

#TRACKING_WINDOWS = 3
TRACKING_WINDOWS = 10
MAX_MISSES = TRACKING_WINDOWS
MIN_HITS = 2

class TrackingCriteria:
    def __init__(self, iou_threshold = 0.3, tracking_windows = TRACKING_WINDOWS, min_hits_frames = math.ceil(TRACKING_WINDOWS/3) + 1, max_missed_frames = TRACKING_WINDOWS, name =""):
        if (max_missed_frames > tracking_windows) or (min_hits_frames > tracking_windows):
            raise ValueError("TrackingCriteria: Error on Window/Hits/Miss")
        if not (0 <= iou_threshold) or (iou_threshold >= 1):
            raise ValueError("TrackingCriteria: Error on 'iou_threshold'")
        self.iou_threshold = iou_threshold
        self.tracking_windows = tracking_windows
        self.min_hits_frames = min_hits_frames
        self.max_missed_frames = max_missed_frames
        self.name = name

    def label(self):
        if (0 != len(self.name)):
            nameStr = f"{self.name}_"
        else:
            nameStr = ""
        label = f"{nameStr}iou_threshold_{self.iou_threshold}__win_{self.tracking_windows}__min_hits_{self.min_hits_frames}__max_misses_{self.max_missed_frames}"
        return label

DEFAULT_TRACKING_CRITERIA = TrackingCriteria(iou_threshold = 0.3, tracking_windows = 3, min_hits_frames = 2, max_missed_frames = 3)

class TrackingStatus(Enum):
    Tracked = auto()
    Tentative = auto()
    ToDelete = auto()

@dataclass
class Track:
    id: int
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    history: deque[bool]
    tracking_criteria: TrackingCriteria = DEFAULT_TRACKING_CRITERIA
    status: TrackingStatus = TrackingStatus.Tentative
    ground_point: tuple[float, float] | None = None
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    qa_nb_confirmations: int = 0
    qa_nb_hits: int = 1


    def qa_confirm_track(self):
        self.status = TrackingStatus.Tracked
        self.qa_nb_confirmations += 1

    def update_status(self):
        min_hits = self.tracking_criteria.min_hits_frames
        max_misses = self.tracking_criteria.max_missed_frames
        hits = sum(self.history)
        misses = len(self.history) - hits

        if self.status == TrackingStatus.Tentative:
            if hits >= min_hits:
                self.qa_confirm_track()
            elif misses >= max_misses:
                self.status = TrackingStatus.ToDelete

        elif self.status == TrackingStatus.Tracked:
            if misses >= max_misses:
                self.status = TrackingStatus.ToDelete

@dataclass
class QA_Track_Record:
    frame_of_creation: int = 0
    frame_of_endOfLife: int = 0
    nb_confirmations: int = 0
    nb_hits: int = 0
    trajectory: list[tuple[float, float]] = field(default_factory=list)


VERY_SHORT_DURATION_CRITERIA = 2
VERY_SHORT_HIT_CRITERIA = 2
class TrackStats(Enum):
    Total_number_created_track = auto()
    Total_number_confirmed_track = auto()
    Total_track_duration = auto()
    Mean_track_duration = auto()
    Median_track_duration = auto()
    Median_nb_hits = auto()
    Hits_distribution = auto()
    Nb_very_short_tracks = auto()
    Nb_maximal_simultaneous_tracks = auto()

@dataclass
class QA_Tracks_Stat:
    # These internal variable stores intermediate values for final statistics computation, stores in 'tracks_stats' dictionary.
    tracks_stats : Dict[str, Any] = field(default_factory=dict)
    tracks_archive : dict = field(default_factory=dict)
    total_number_created_track: int = 0
    total_number_confirmed_track: int = 0
    total_track_duration: int = 0
    mean_track_duration: float = 0
    median_track_duration: float = 0
    median_nb_hits = 0
    hits_distribution: list[int] = field(default_factory=list)
    nb_very_short_tracks: int = 0
    nb_maximal_simultaneous_tracks: int = 0

    very_short_duration_level = VERY_SHORT_DURATION_CRITERIA
    very_short_hit_criteria = VERY_SHORT_HIT_CRITERIA
    NEW_TRACK_RECORD_IDX = 0
    DEL_TRACK_RECORD_IDX = 1

    def recordNewTrack(self, id, frame_nb):
        self.tracks_archive[id] = QA_Track_Record(frame_of_creation = frame_nb)


    def updateTrackEOL(self, track, frame_nb):
        record = self.tracks_archive[track.id]

        record.frame_of_endOfLife = frame_nb
        record.nb_confirmations = track.qa_nb_confirmations
        record.nb_hits = track.qa_nb_hits
        record.trajectory = track.trajectory.copy()


    def processTrackingArchives(self):
        tracks_creation_history = {}
        # Reset statistics: makes this function idempotent
        self.total_number_created_track = len(self.tracks_archive)
        self.total_number_confirmed_track = 0
        self.total_track_duration = 0
        self.mean_track_duration = 0
        self.median_track_duration = 0
        self.median_nb_hits = 0
        self.hits_distribution = []
        self.nb_very_short_tracks = 0
        self.nb_maximal_simultaneous_tracks = 0

        if self.total_number_created_track == 0:
            return

        durations = []

        for record in self.tracks_archive.values():
            duration = (
                record.frame_of_endOfLife
                - record.frame_of_creation
                + 1
            )

            durations.append(duration)

            self.total_number_confirmed_track += record.nb_confirmations
            self.total_track_duration += duration

            if record.nb_hits <= self.very_short_hit_criteria:
                self.nb_very_short_tracks += 1

            creation_time = record.frame_of_creation
            eol_time = record.frame_of_endOfLife

            if creation_time not in tracks_creation_history:
                tracks_creation_history[creation_time] = [0, 0]

            if eol_time not in tracks_creation_history:
                tracks_creation_history[eol_time] = [0, 0]

            tracks_creation_history[creation_time][
                self.NEW_TRACK_RECORD_IDX
            ] += 1

            tracks_creation_history[eol_time][
                self.DEL_TRACK_RECORD_IDX
            ] += 1

        # Duration statistics
        self.mean_track_duration = (
            self.total_track_duration
            / self.total_number_created_track
        )

        self.median_track_duration = median(durations)

        # Maximum number of simultaneous tracks
        nb_simultaneous_tracks = 0

        for event_time in sorted(tracks_creation_history):
            nb_simultaneous_tracks += (
                tracks_creation_history[event_time][self.NEW_TRACK_RECORD_IDX]
                - tracks_creation_history[event_time][self.DEL_TRACK_RECORD_IDX]
            )

            self.nb_maximal_simultaneous_tracks = max(
                self.nb_maximal_simultaneous_tracks,
                nb_simultaneous_tracks
            )

        hits = [
            record.nb_hits
            for record in self.tracks_archive.values()
            ]
        self.hits_distribution = Counter(hits)
        self.median_nb_hits = median(hits)

        # Sanity checks
        assert self.total_track_duration >= self.total_number_created_track
        assert self.mean_track_duration >= 1
        assert (
            self.total_number_confirmed_track
            <= self.total_number_created_track
        )
        assert (
            self.nb_very_short_tracks
            <= self.total_number_created_track
        )
        self.tracks_stats[TrackStats.Total_number_created_track] = self.total_number_created_track
        self.tracks_stats[TrackStats.Total_number_confirmed_track] = self.total_number_confirmed_track
        self.tracks_stats[TrackStats.Total_track_duration] = self.total_track_duration
        self.tracks_stats[TrackStats.Mean_track_duration] = self.mean_track_duration
        self.tracks_stats[TrackStats.Median_track_duration] = self.median_track_duration
        self.tracks_stats[TrackStats.Median_nb_hits] = self.median_nb_hits
        self.tracks_stats[TrackStats.Hits_distribution] = self.hits_distribution
        self.tracks_stats[TrackStats.Nb_very_short_tracks] = self.nb_very_short_tracks
        self.tracks_stats[TrackStats.Nb_maximal_simultaneous_tracks] = self.nb_maximal_simultaneous_tracks


    def displayTracksStats(self):
        print ("Total number created track = ", self.tracks_stats[TrackStats.Total_number_created_track])
        print ("Total number confirmed track = ", self.tracks_stats[TrackStats.Total_number_confirmed_track])
        print ("Total track duration = ", self.tracks_stats[TrackStats.Total_track_duration])
        print ("Mean_track_duration = ", self.tracks_stats[TrackStats.Mean_track_duration])
        print ("Median track duration = ", self.tracks_stats[TrackStats.Median_track_duration])
        print ("Median nb hits = ", self.tracks_stats[TrackStats.Median_nb_hits])
        print ("Hits distribution = ")
        print(sorted(self.hits_distribution.items()))
        print ("nb very short tracks = ", self.tracks_stats[TrackStats.Nb_very_short_tracks])
        print ("nb maximal simultaneous tracks = ", self.tracks_stats[TrackStats.Nb_maximal_simultaneous_tracks])

class IoUTracker:
    def __init__(self, tracking_criteria = DEFAULT_TRACKING_CRITERIA):
        self.tracks = []
        self.frame_number = 0
        self.iou_threshold = tracking_criteria.iou_threshold
        self.max_missed_frames = tracking_criteria.max_missed_frames
        self.next_track_id = 0
        self.trajectory_archive = dict()
        self.qa_tracks_analyze : QA_Tracks_Stat = QA_Tracks_Stat()
        self.tracking_criteria = tracking_criteria


    def qa_nb_created_track(self):
        return self.next_track_id


    def qa_update_EOL(self):
        for track in self.tracks:
            self.qa_tracks_analyze.updateTrackEOL(track, self.frame_number)


    def update_get_tracks_stat(self):
        self.qa_update_EOL()
        self.qa_tracks_analyze.processTrackingArchives()
        return self.qa_tracks_analyze.tracks_stats


    def create_track(self, detection):
        track = Track(
            id=self.next_track_id,
            bbox=detection[IDX_DET_BBOX],
            class_name=detection[IDX_DET_CLASS],
            confidence=detection[IDX_DET_CONFIDENCE],
            history=deque(
                [True],
                maxlen=self.tracking_criteria.tracking_windows
                ),
            tracking_criteria=self.tracking_criteria,
            )
        x1, y1, x2, y2 = track.bbox
        ground_point = ((x1 + x2) / 2, y2)
        track.ground_point = ground_point
        track.trajectory.append(ground_point)

        self.qa_tracks_analyze.recordNewTrack(self.next_track_id, self.frame_number)
        self.next_track_id += 1
        self.tracks.append(track)


    def initTrackFromDet(self, detections):
        count_track = 0
        for det in detections:
            self.create_track(det)

    def displayTracks(self):
        for track in self.tracks:
            print("Id = ", track.id, " track.class_name = ", track.class_name, " track.confidence = ", track.confidence, " track.bbox = ", track.bbox, " track.status = ", track.status)

    def archive(self, track):
        self.trajectory_archive[track.id] = track.trajectory

    def update(self, detections):
        # No reference for beginning = Each detections seems valid
        self.frame_number += 1
        if not self.tracks:
            self.initTrackFromDet(detections)
            return self.tracks
        bbox_tracks = [track.bbox for track in self.tracks]
        bbox_det = [det[IDX_DET_BBOX] for det in detections]
        IoUMatrix = compute_iou_matrix(bbox_tracks, bbox_det)
        optimal_tracks = optimal_match_from_score_matrix(IoUMatrix, self.iou_threshold)
        matched_refs = set()
        matched_dets = set()
        for opt_track in optimal_tracks:
            score, idx_ref, idx_det = opt_track
            matched_refs.add(idx_ref)
            matched_dets.add(idx_det)
            self.tracks[idx_ref].confidence = detections[idx_det][IDX_DET_CONFIDENCE]
            self.tracks[idx_ref].class_name = detections[idx_det][IDX_DET_CLASS]
            self.tracks[idx_ref].bbox = detections[idx_det][IDX_DET_BBOX]
            self.tracks[idx_ref].history.append(True)     # matched
            self.tracks[idx_ref].qa_nb_hits += 1
            self.tracks[idx_ref].update_status()
            x1, y1, x2, y2 = self.tracks[idx_ref].bbox
            ground_point = ((x1 + x2) / 2, y2)
            self.tracks[idx_ref].ground_point = ground_point
            self.tracks[idx_ref].trajectory.append(ground_point)
        for idx_track in range(len(self.tracks)):
            if not (idx_track in matched_refs):
                self.tracks[idx_track].history.append(False)
                self.tracks[idx_track].update_status()
        is_to_be_deleted = lambda x : (TrackingStatus.ToDelete == x.status)
        keep, toBeArchived = more_itertools.partition(is_to_be_deleted, self.tracks)
        toKeep = list(keep) #Will be removed
        toDelete = list(toBeArchived)
        self.tracks = toKeep
        for idx_det, detection in enumerate(detections):
            if idx_det not in matched_dets:
                self.create_track(detection)
        for track in toDelete:
            self.qa_tracks_analyze.updateTrackEOL(track, self.frame_number)
