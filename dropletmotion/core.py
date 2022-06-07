"""
This module gives tools to extract the position of droplets moving across a
video frame

Classes
-------
DropletTrack
    Load video source, detect positions and carry out analysis

Functions
---------
drop_profile
    Analyse droplet profile and return advancing and receding position
"""

# Imports
import pathlib
import re
import logging
import numpy as np
import cv2
import pandas as pd
import scipy.signal
from dropletmotion import utility


# Assign logger
log = logging.getLogger(__name__)


class DropletTrack():
    """
    Class that deals with analysis of video
    file to extract droplet position.

    Attributes
    ----------
    video : cv2.VideoCapture
        Video to be analysed
    file_folder : pathlib.Path
        Folder containing source file. Path is relative to the current
        working directory
    file_name : str
        File name of source file
    x : list
        List of ndarray each containing droplet position 'x' for given time
        't'. It can include also positions of the advancing and receding
        fronts, respectively 'x_adv' and 'x_rec'.
    drop_list : list
        List of int with detected droplets. Takes into account missing or
        skipped droplets
    sample_info : dict
        Dictionary containing sample information extracted from file_name
        via the utility.extract_info function

    Methods
    -------
    droplet_detect
        Analyze loaded video and extracts detected object position
    motion_save
        Save csv output of detected object position
    """

    def __init__(
        self, video_path, ref_mm=25, ref_px_err=5, ref_mm_err=0.05) -> None:
        """
        Load video and initialize method attributes.

        Parameters
        ----------
        video_path : str
            Path to video file
        ref_mm : float
            Length of reference object in mm (default=2.9)
        ref_px_err : float
            Rrror on ref_px (default=1)
        ref_mm_err: float
            Error on ref_mm (default=0.02)
        """

        self.video = cv2.VideoCapture(str(video_path))

        # Video and sample properties used throughout class
        self.props = {
            'file_folder': pathlib.Path(video_path).parent,
            'file_name': pathlib.Path(video_path).stem,
            **utility.extract_info(
                pathlib.Path(video_path).stem, ref_mm, ref_px_err, ref_mm_err
            ),
            'frame_width': int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'frame_height': int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

        log.info('Current video: %s', self.props['file_name'])

        # Initialize remaining class attributes
        self.x_drop = pd.DataFrame({})

        # Initialize private helper attributes
        self.__x_det = []
        self.__tracker = []
        self.__detect_params = {}

        # Check missed or skipped droplets
        def __extract_miss_skip(key):
            """ Extract list of missed or skippe droplets """
            try:
                return [
                    int(i) for i in re.findall(
                        rf'{key}\S*', self.props['file_name']
                    )[0].replace(key,'').split('-')
                ]
            except IndexError:
                return []

        self.__miss_skip_drop = {
            'miss': __extract_miss_skip('miss'),
            'skip': __extract_miss_skip('skip')
        }


    def droplet_detect(
        self, detect_roi=0.2, correl_th=0.7,
        black_value=20, fronts_calc=False,
        needle_check=True, needle_height_th=4
        ) -> bool:
        """
        Method that analyses the loaded video detecting and tracking
        droplets moving across the frame. Template object is selected
        from first frame of video.
        NOTE: droplets must enter frame from the left.

        Parameters
        ----------
        detect_roi : float
            Fraction of frame used for detection (default=0.2)
        correl_th : float
            Threshold for max normed correlation coefficient used to determine
            success of tracking (default=0.7, max=1)
        black_value : int
            Initial guess for greyscale value of black (default=20)
        fronts_calc : bool
            Flag for frame by frame extraction of advancing and receding fronts
            position (default=False)
        needle_check : bool
            Flag for checking of whether needle (or generally black
            space on top of droplet) is in frame (default=True)
        needle_height_th : int
            Pixel threshold from the top edge to detect wheter needle has left
            frame (or droplet is not still springing back from detachment)
            (default=4)

        Returns
        -------
        break_flag : bool
            Flag that signals if operation has been interrupted (defaul=False)
        """

        # Pass arguments to class attribute for easier access by helper methods
        self.__detect_params = {
            'detect_roi': detect_roi,
            'correl_th': correl_th,
            'black_value': black_value,
            'fronts_calc': fronts_calc,
            'needle_check': needle_check,
            'needle_height_th': needle_height_th,
            'detect_width': max(
                round(detect_roi * self.props['frame_width']), 200
            ),
            'drift_warning': False
        }

        # Initialize method parameters
        break_flag = False
        detection = True

        # Turn on base radius calculation if dealing with crossing experiment
        if 'exp_C' in self.props['file_name']:
            self.__detect_params['fronts_calc'] = True

        # Detect droplet template
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        break_flag = self.__detect_template()

        while True:
            can_read, frame = self.video.read()

            # Set conditions for break cycle
            if not can_read or break_flag:
                break

            # Convert frame to grayscale
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Skip frame if empty or if needle is in frame
            if (
                self.__empty_frame(frame_bw) or
                self.__needle_in_frame(frame_bw)
            ):
                if True in [i['status'] for i in self.__tracker]:
                    log.error('Skipped frame with tracked droplet')
                    break_flag = True
                    break
                continue

            # Draw line at detection region
            frame_out = frame.copy()
            cv2.line(
                frame_out, (self.__detect_params['detect_width'], 0),
                (
                    self.__detect_params['detect_width'],
                    self.props['frame_height'] + 1
                ), (80, 43, 229), 2
            )

            # Tracking loop
            break_flag, detection, frame_out = self.__tracking_loop(
                frame_bw, frame_out
            )

            if break_flag:
                break

            # Detection of new droplets
            if detection:
                detection = self.__detection_routine(frame_bw)

            # Output analysed video
            cv2.imshow(
                'Detected droplet - ' + self.props['file_name'], frame_out
            )

            # This command ensures that video stops correctly, or breaks with
            # the 'q' key.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break_flag = True
                break

        # Releases video file and closes video window
        self.video.release()
        cv2.destroyAllWindows()

        # Wrap up analysis and save output
        if not break_flag:

            self.x_drop = (
                pd.DataFrame(self.__x_det)
                .query("droplet != @self._DropletTrack__miss_skip_drop['skip']")
                .sort_values(by=['droplet', 't'])
                .reset_index(drop=True)
            )

            # Log warnings
            self.__volume_warning()
            if self.__detect_params['drift_warning']:
                log.warning('Possible tracker drift detected')

            # Format time-position dataframe for .csv export
            break_flag = self.motion_save()

        # Return method parameters that can be used by external scripts
        return break_flag

    def motion_save(self) -> bool:
        """
        Callable function to write csv output of detected position
        and its relative scale error with the appropriate format

        Returns
        -------
        break_flag : bool
            Flag that signals if operation was not brought to completion.
        """

        # Determine if 'Extracted data' sub-folder exists and create it if not
        save_folder = self.props['file_folder'] / 'Extracted data'
        save_folder.mkdir(exist_ok=True)

        if not self.x_drop.empty:
            save_name = self.props['file_name'] + ' position.csv'
            self.x_drop.to_csv(save_folder / save_name, index=False)
            return False

        log.error('No droplets detected.')
        return True

    def __empty_frame(self, frame_bw) -> bool:
        """ Check if frame is empty by imposing no black pixels in midline """
        return bool(
            np.amin(
                frame_bw[int(self.props['frame_height'] / 2), :]
            ) >= self.__detect_params['black_value'] + 120
        )

    def __needle_in_frame(self, frame_bw) -> bool:
        """ Check if needle is still in frame """
        if self.__detect_params['needle_check']:
            return bool(
                np.amin(
                    frame_bw[self.__detect_params['needle_height_th'], :]
                ) <= self.__detect_params['black_value'] + 20
            )
        return False

    def __droplet_in_frame(self, frame_bw) -> bool:
        """ Check if droplet is fully in frame """
        return bool(
            np.all(
                frame_bw[:int(self.props['frame_height'] / 3 * 2), 0:10] > 120
            )
        )

    def __detection_switch(self, bbox_left_x) -> bool:
        """ Return false if the matched bbox is right of detection area """
        return bool(bbox_left_x > self.__detect_params['detect_width'])

    def __detect_template(self) -> bool:
        """
        Detect first droplet to use as template for the detection of further
        droplets by template matching. The template is stored in
        self.__detect_params['template'].

        The method also updates the initial black value with the one extracted
        from the template.

        Returns
        -------
        break_flag : bool
            Flag that signals interruption of the process.
        """

        template_skip_frames = 0
        break_flag = False

        while True:
            can_read, frame = self.video.read()

            if not can_read:
                log.error('Cannot read video file')
                break_flag = True
                break

            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Skip frames before first droplet is in full view and needle is
            # out of picture
            if (
                not self.__empty_frame(frame_bw) and
                self.__droplet_in_frame(frame_bw) and
                not self.__needle_in_frame(frame_bw)
            ):
                # Skip one frame for safety
                if template_skip_frames < 1:
                    template_skip_frames += 1
                else:
                    break

        if not break_flag:
            try:
                _, frame_th = cv2.threshold(
                    frame_bw, 0, 255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                cont, _ = cv2.findContours(
                    frame_th[:, :self.__detect_params['detect_width']],
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                # Find largest contour and calculate bounding box
                bbox_temp = cv2.boundingRect(max(cont, key=cv2.contourArea))
                self.__detect_params['template'] = frame_bw[
                    bbox_temp[1]:bbox_temp[1] + bbox_temp[3],
                    bbox_temp[0]:bbox_temp[0] + bbox_temp[2]
                ]

                (
                    self.__detect_params['template_width'],
                    self.__detect_params['template_height']
                ) = self.__detect_params['template'].shape[::-1]

            except ValueError:
                break_flag = True
                log.error('No suitable droplet detected for template')

            # Redefine black value with more precise value
            self.__detect_params['black_value'] = np.amin(
                frame_bw[int(self.props['frame_height'] / 2), :]
            )

        return break_flag

    def __tracking_loop(self, frame_bw, frame_out) -> tuple[bool, bool, np.array]:
        """
        Perform tracking by looping through active tracking objects.

        Parameters
        ----------
        frame_bw : np.array
            Grayscale video frame to analyse.
        frame_out : np.array
            Frame on which to draw contours and center of mass.

        Returns
        -------
        break_flag : bool
            Flag that signals interruption of the process.
        detection : bool
            Flag that signals whether to start detection routine.
        frame_out : np.array
            Updated frame with drawn contours.
        """

        # Initialize parameters
        detection = False
        break_flag = False

        # Binarize frame with Otsu's threshold
        _, frame_th = cv2.threshold(
            frame_bw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Tracking loop
        for tracker in self.__tracker:

            # Check tracker activity
            if tracker['status']:
                # Update tracker
                track_success, bbox_tr = tracker['tracker'].update(frame_bw)

                # Redefine bounding box as tuple of integers, expandig it
                # horizontally to account for droplet elongation and
                # optical errors and using full vertical span
                bbox_obj = (
                    max(
                        int(
                            bbox_tr[0] + bbox_tr[2] / 2
                            - 1.5 * self.__detect_params['template_width']),
                        0),
                    0,
                    int(self.__detect_params['template_width'] * 2.5),
                    self.props['frame_height']
                )

                if track_success:

                    # Detect center of mass of droplet in tracked region
                    drop_roi = frame_th[
                        bbox_obj[1]:bbox_obj[1] + bbox_obj[3],
                        bbox_obj[0]:bbox_obj[0] + bbox_obj[2]
                    ]

                    # Find contours
                    try:
                        frame_out =  self.__update_position(
                            tracker, drop_roi, bbox_obj, frame_out
                        )
                    except (ValueError, ZeroDivisionError):
                        break_flag = True
                        log.error('Droplet contour error')
                        return break_flag, detection, frame_out

                    detection = self.__detection_switch(bbox_obj[0])

                # Re-initialize tracker if it does not track object
                # right after detection, and only if tracker is the
                # the most recent active.
                elif (
                    'volume' not in tracker and
                    tracker['fail_count'] <= 5 and
                    tracker['droplet'] == max(
                        i['droplet'] for i in self.__tracker
                    )
                ):
                    del tracker['tracker']
                    tracker['fail_count'] += 1
                    detection = True
                    log.warning(
                        'Tracker %d failed. Re-initializing',
                        tracker['droplet']
                    )

                # Stop analysis if too many tracking errors
                else:
                    break_flag = True
                    log.error(
                        str(
                            'Incorrect droplet tracking. ' +
                            'Tracker %d failed. Analysis stopped'),
                        tracker['droplet']
                    )
                    return break_flag, detection, frame_out

        # Turn on detection if there are no active trackers (failsafe)
        if True not in [i['status'] for i in self.__tracker]:
            detection = True

        return break_flag, detection, frame_out

    def __update_position(self, tracker, drop_roi, bbox_obj, frame_out) -> np.array:
        """
        Calculate tracked droplet position and update attributes.

        Parameters
        ----------
        tracker : dict
            Item of self.__trackers, with information about currently active
            tracker object.
        drop_roi : np.array
            Portion of thresholded frame with tracked droplet.
        bbox_obj : np.array
            Bounding box of the region with the tracked droplet, containing the
            coordinates of drop_roi relative to the entire frame.
        frame_out : np.array
            Frame on which to draw contours and center of mass.

        Returns
        -------
        break_flag : bool
            Flag that signals interruption of the process.
        frame_out : np.array
            Updated frame with drawn contours.
        """

        # Color for contours
        blue = (200, 120, 70)

        cont, _ = cv2.findContours(
            drop_roi, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
            offset=(bbox_obj[0], bbox_obj[1])
        )

        # Find largest contour and calculate convex hull
        cont_max = max(cont, key=cv2.contourArea)
        hull = cv2.convexHull(cont_max)
        bbox_hull = cv2.boundingRect(hull)

        # Check if droplet is still in still fully in frame
        if bbox_hull[0] + bbox_hull[2] < self.props['frame_width']:

            # Check if tracker is drifting away from object by checking for
            # collision between detected object and left edge of max object
            # bounding box
            self.__detect_params['drift_warning'] = any(
                [item[0][0] for item in hull <= bbox_obj[0]] +
                [self.__detect_params['drift_warning']]
            )

            # Calculate and save center of mass
            moments = cv2.moments(hull)
            cmx = moments["m10"] / moments["m00"]
            cmy = moments["m01"] / moments["m00"]

            self.__x_det.append({
                'droplet': tracker['droplet'],
                't': self.video.get(cv2.CAP_PROP_POS_FRAMES) / self.props['fps'],
                'x': cmx * self.props['scale']
            })

            # Calculate drop volume
            if 'volume' not in tracker:
                _, _, tracker['volume'] = drop_profile(
                    cont_max, self.props['scale'], True
                )

            # Calculate base radius
            if self.__detect_params['fronts_calc']:
                (
                    self.__x_det[-1]['x_rec'], self.__x_det[-1]['x_adv'], _
                ) = drop_profile(
                    cont_max, self.props['scale']
                )

            # Annotate displayed frame
            cv2.putText(
                frame_out, f"{tracker['droplet']}",
                (
                    bbox_hull[0] + bbox_hull[2] + 5,
                    int(bbox_hull[1] + bbox_hull[3] / 2)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, blue, 2
            )
            cv2.circle(frame_out, (int(cmx), int(cmy)), 7, blue, -1)
            cv2.drawContours(frame_out, [hull], -1, blue,
                                2)
        else:
            # Stop tracker
            tracker['status'] = False

        return frame_out

    def __detection_routine(self, frame_bw) -> bool:
        """
        Perform droplet detection by template matching with normalized
        correlation coefficient and initialize tracker in case of success.

        Parameters
        ----------
        frame_bw : np.array
            Grayscale video frame to analyse.

        Returns
        -------
        detection : bool
            Flag that signals whether to start detection routine.
        """

        detection = True

        # Slice image to detection area
        frame_det = frame_bw[:, :self.__detect_params['detect_width']]

        # Calculate matching and extract maximum correlation
        result = cv2.matchTemplate(
            frame_det, self.__detect_params['template'], cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Matching success
        if max_val > self.__detect_params['correl_th']:

            if self.__tracker and 'tracker' not in self.__tracker[-1]:
                # Re-create tracker if it failed
                self.__tracker[-1]['tracker'] = cv2.TrackerCSRT_create()
            else:

                # Update droplet number
                try:
                    drop_count = self.__tracker[-1]['droplet'] + 1
                except IndexError:
                    drop_count = 1
                if drop_count in self.__miss_skip_drop['miss']:
                    while drop_count in self.__miss_skip_drop['miss']:
                        drop_count += 1

                # Create new tracker
                self.__tracker.append({
                    'tracker': cv2.TrackerCSRT_create(),
                    'status': True,
                    'droplet': drop_count,
                    'fail_count': 0
                })

            # Build bounding box for detected object from correlation
            # maxima position
            top_left_w = max_loc[0]
            top_left_h = max_loc[1]
            bbox_det = (
                top_left_w, top_left_h,
                self.__detect_params['template_width'],
                self.__detect_params['template_height']
            )

            # Initialize tracking of detected object
            self.__tracker[-1]['tracker'].init(frame_bw, bbox_det)

            # Failsafe to avoid double detection if tracker fails
            # immediately after detection
            detection = False

        return detection

    def __volume_warning(self) -> None:
        """ Give warning if detected droplets are smaller than average """

        mean_volume = np.nanmean(np.array([
            i['volume'] for i in self.__tracker
        ]))

        for item in self.__tracker:
            if item['volume'] <= 0.8 * mean_volume:
                log.warning(
                    'Droplet %d too small. Possible dosing mistake',
                    item['droplet']
                )


def drop_profile(contour, scale, volume_check=False) -> tuple[float, float, float]:
    """
    Analyse drop profile and return left and right base radius,
    relative to droplet center of mass. Also calculate droplet
    volume in uL if the volume_check option is active

    Parameter
    ---------
    contour : np.array
        Numpy array with the contour detected by cv2.findContours
    scale : float
        Image scale, in mm / px
    volume_check : bool
        Flag for calculation of the droplet volume (default=False)

    Returns
    -------
    x_left : float
        Position of the left contact point between droplet and surface, in mm
    x_right : float
        Position of the right contact point between droplet and surface, in mm
    volume : float
        Droplet volume in µL
    """

    # Flatten contour array to a more manageable form
    cont_array = np.array([item[0] for item in contour])
    cont_height = np.arange(
        cont_array[:, 1].min(), cont_array[:, 1].max() + 1)

    # Extract profiles by removing duplicates and interior points
    right = np.fromiter(
        (max(cont_array[cont_array[:, 1]==i, 0]) for i in cont_height),
        dtype=np.int32)
    left = np.fromiter(
        (min(cont_array[cont_array[:, 1]==i, 0]) for i in cont_height),
        dtype=np.int32)

    # Determine baseline from right advancing profile
    try:
        base_index = scipy.signal.find_peaks(-right)[0][0]
    except IndexError:
        base_index = -1

    x_right = right[base_index] * scale
    x_left = left[base_index] * scale

    if volume_check:
        # Remove contour below baseline
        base_height = cont_height[base_index]
        cont_drop = cont_array[cont_array[:, 1]<=base_height]

        hull = cv2.convexHull(cont_drop)
        volume = (cv2.contourArea(hull) ** 1.5 * scale ** 3)
    else:
        volume = np.nan

    return x_left, x_right, volume
