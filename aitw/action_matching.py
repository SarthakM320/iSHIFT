'''
Adapted from https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/action_matching.py
Adapted from https://github.com/google-research/google-research/tree/master/android_in_the_wild
'''

import re
import numpy as np

from action_type import ActionType


_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4

# Interval determining if an action is a tap or a swipe.
_SWIPE_DISTANCE_THRESHOLD = 0.04


count = 0


def _yx_in_bounding_boxes(
    yx, bounding_boxes
):
  """Check if the (y,x) point is contained in each bounding box.

  Args:
    yx: The (y, x) coordinate in pixels of the point.
    bounding_boxes: A 2D int array of shape (num_bboxes, 4), where each row
      represents a bounding box: (y_top_left, x_top_left, box_height,
      box_width). Note: containment is inclusive of the bounding box edges.

  Returns:
    is_inside: A 1D bool array where each element specifies if the point is
      contained within the respective box.
  """
  y, x = yx

  # `bounding_boxes` has shape (n_elements, 4); we extract each array along the
  # last axis into shape (n_elements, 1), then squeeze unneeded dimension.
  top, left, height, width = [
      np.squeeze(v, axis=-1) for v in np.split(bounding_boxes, 4, axis=-1)
  ]

  # The y-axis is inverted for AndroidEnv, so bottom = top + height.
  bottom, right = top + height, left + width

  return np.logical_and(y >= top, y <= bottom) & np.logical_and(
      x >= left, x <= right)


def _resize_annotation_bounding_boxes(
    annotation_positions, annotation_width_augment_fraction,
    annotation_height_augment_fraction):
  """Resize the bounding boxes by the given fractions.

  Args:
    annotation_positions: Array of shape (N, 4), where each row represents the
      (y, x, height, width) of the bounding boxes.
    annotation_width_augment_fraction: The fraction to augment the box widths,
      E.g., 1.4 == 240% total increase.
    annotation_height_augment_fraction: Same as described for width, but for box
      height.

  Returns:
    Resized bounding box.

  """
  height_change = (
      annotation_height_augment_fraction * annotation_positions[:, 2])
  width_change = (
      annotation_width_augment_fraction * annotation_positions[:, 3])

  # Limit bounding box positions to the screen.
  resized_annotations = np.stack([
      np.maximum(0, annotation_positions[:, 0] - (height_change / 2)),
      np.maximum(0, annotation_positions[:, 1] - (width_change / 2)),
      np.minimum(1, annotation_positions[:, 2] + height_change),
      np.minimum(1, annotation_positions[:, 3] + width_change),
  ],
                                  axis=1)
  return resized_annotations


def is_tap_action(normalized_start_yx,
                  normalized_end_yx):
  distance = np.linalg.norm(
      np.array(normalized_start_yx) - np.array(normalized_end_yx))
  return distance <= _SWIPE_DISTANCE_THRESHOLD


def _is_non_dual_point_action(action_type):
  return np.not_equal(action_type, ActionType.DUAL_POINT)


def _check_tap_actions_match(
    tap_1_yx,
    tap_2_yx,
    annotation_positions,
    matching_tap_distance_threshold_screen_percentage,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
):
  """Determines if two tap actions are the same."""
  try:
    resized_annotation_positions = _resize_annotation_bounding_boxes(
        annotation_positions,
        annotation_width_augment_fraction,
        annotation_height_augment_fraction,
    )
    
    # Check if the ground truth tap action falls in an annotation's bounding box.
    tap1_in_box = _yx_in_bounding_boxes(tap_1_yx, resized_annotation_positions)
    tap2_in_box = _yx_in_bounding_boxes(tap_2_yx, resized_annotation_positions)
    both_in_box = np.max(tap1_in_box & tap2_in_box)
  except Exception as e:
    # print(f'Error: {e}, {annotation_positions.shape}')
    both_in_box = False

  # If the ground-truth tap action falls outside any of the annotation
  # bounding boxes or one of the actions is inside a bounding box and the other
  # is outside bounding box or vice versa, compare the points using Euclidean
  # distance.
  within_threshold = (
      np.linalg.norm(np.array(tap_1_yx) - np.array(tap_2_yx))
      <= matching_tap_distance_threshold_screen_percentage
  )
  return np.logical_or(both_in_box, within_threshold)


def _check_drag_actions_match(
    drag_1_touch_yx,
    drag_1_lift_yx,
    drag_2_touch_yx,
    drag_2_lift_yx,
):
  """Determines if two drag actions are the same."""
  # Store drag deltas (the change in the y and x coordinates from touch to
  # lift), magnitudes, and the index of the main axis, which is the axis with
  # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
  # ending at (0.3, 0.5) has a main axis index of 1).
  drag_1_deltas = drag_1_lift_yx - drag_1_touch_yx
  drag_1_magnitudes = np.abs(drag_1_deltas)
  drag_1_main_axis = np.argmax(drag_1_magnitudes)
  drag_2_deltas = drag_2_lift_yx - drag_2_touch_yx
  drag_2_magnitudes = np.abs(drag_2_deltas)
  drag_2_main_axis = np.argmax(drag_2_magnitudes)

  return np.equal(drag_1_main_axis, drag_2_main_axis)


def check_actions_match(
    action_1_touch_yx,
    action_1_lift_yx,
    action_1_action_type,
    action_2_touch_yx,
    action_2_lift_yx,
    action_2_action_type,
    annotation_positions,
    tap_distance_threshold = _TAP_DISTANCE_THRESHOLD,
    annotation_width_augment_fraction = ANNOTATION_WIDTH_AUGMENT_FRACTION,
    annotation_height_augment_fraction = ANNOTATION_HEIGHT_AUGMENT_FRACTION,
):
  """Determines if two actions are considered to be the same.

  Two actions being "the same" is defined here as two actions that would result
  in a similar screen state.

  Args:
    action_1_touch_yx: The (y, x) coordinates of the first action's touch.
    action_1_lift_yx: The (y, x) coordinates of the first action's lift.
    action_1_action_type: The action type of the first action.
    action_2_touch_yx: The (y, x) coordinates of the second action's touch.
    action_2_lift_yx: The (y, x) coordinates of the second action's lift.
    action_2_action_type: The action type of the second action.
    annotation_positions: The positions of the UI annotations for the screen. It
      is A 2D int array of shape (num_bboxes, 4), where each row represents a
      bounding box: (y_top_left, x_top_left, box_height, box_width). Note that
      containment is inclusive of the bounding box edges.
    tap_distance_threshold: The threshold that determines if two taps result in
      a matching screen state if they don't fall the same bounding boxes.
    annotation_width_augment_fraction: The fraction to increase the width of the
      bounding box by.
    annotation_height_augment_fraction: The fraction to increase the height of
      of the bounding box by.

  Returns:
    A boolean representing whether the two given actions are the same or not.
  """
  action_1_touch_yx = np.asarray(action_1_touch_yx)
  action_1_lift_yx = np.asarray(action_1_lift_yx)
  action_2_touch_yx = np.asarray(action_2_touch_yx)
  action_2_lift_yx = np.asarray(action_2_lift_yx)

  # Checks if at least one of the actions is global (i.e. not DUAL_POINT),
  # because if that is the case, only the actions' types need to be compared.
  has_non_dual_point_action = np.logical_or(
      _is_non_dual_point_action(action_1_action_type),
      _is_non_dual_point_action(action_2_action_type),
  )

  different_dual_point_types = np.logical_xor(
      is_tap_action(action_1_touch_yx, action_1_lift_yx),
      is_tap_action(action_2_touch_yx, action_2_lift_yx),
  )

  is_tap = np.logical_and(
      is_tap_action(action_1_touch_yx, action_1_lift_yx),
      is_tap_action(action_2_touch_yx, action_2_lift_yx),
  )

  taps_match = _check_tap_actions_match(
      action_1_touch_yx,
      action_2_touch_yx,
      annotation_positions,
      tap_distance_threshold,
      annotation_width_augment_fraction,
      annotation_height_augment_fraction,
  )

  taps_match = np.logical_and(is_tap, taps_match)

  drags_match = _check_drag_actions_match(
      action_1_touch_yx, action_1_lift_yx, action_2_touch_yx, action_2_lift_yx
  )
  drags_match = np.where(is_tap, False, drags_match)

  return np.where(
      has_non_dual_point_action,
      np.equal(action_1_action_type, action_2_action_type),
      np.where(
          different_dual_point_types,
          False,
          np.logical_or(taps_match, drags_match),
      ),
  )


def action2json(step_data):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
        else:
            if step_data["action_type_text"] == 'scroll down':
                touch_point = [0.5, 0.8]
                lift_point = [0.5, 0.2]
            elif step_data["action_type_text"] == 'scroll up':
                touch_point = [0.5, 0.2]
                lift_point = [0.5, 0.8]
            elif step_data["action_type_text"] == 'scroll left':
                touch_point = [0.2, 0.5]
                lift_point = [0.8, 0.5]
            elif step_data["action_type_text"] == 'scroll right':
                touch_point = [0.8, 0.5]
                lift_point = [0.2, 0.5]
    else:
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]

    if action_type == 3:
        typed_text = step_data["type_text"]
    else:
        typed_text = ""

    action = {"action_type": action_type, "touch_point": touch_point, "lift_point": lift_point,
              "typed_text": typed_text}

    action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    if action["typed_text"] is not None:
        action["typed_text"] = action["typed_text"].lower()
    return action
