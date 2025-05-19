# 这个脚本是供Grasshopper调用的，用于分析truss的图像

import cv2
import numpy as np

# Read the hand-drawn sketch
input_image_path = 'D:/Rene/OpenCV/img/test4.jpg'
# output_image_path = os.path.join(output_dir, 'output_opencv_0415_1.png') # Removed image output path
image = cv2.imread(input_image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite(os.path.join(output_dir, 'step1_gray.png'), gray) # Removed image output

# Use Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
# cv2.imwrite(os.path.join(output_dir, 'step2_blurred.png'), blurred) # Removed image output

# Adaptive thresholding to improve edge detection quality
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imwrite(os.path.join(output_dir, 'step3_thresh.png'), thresh) # Removed image output

# Invert the binary image
edges = cv2.bitwise_not(thresh)
# cv2.imwrite(os.path.join(output_dir, 'step4_bitwise_not.png'), edges) # Removed image output (was already commented)

# Add erosion operation to thin lines
kernel = np.ones((3,3), np.uint8)
eroded_edges = cv2.erode(edges, kernel, iterations=1)
# cv2.imwrite(os.path.join(output_dir, 'step4_eroded.png'), eroded_edges) # Removed image output

# Use Hough Transform to detect line segments, adjust parameters for accuracy
lines = cv2.HoughLinesP(eroded_edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=30)

# Define line segment merging function
def merge_lines(lines, angle_threshold=10, parallel_distance_threshold=30):
    """
    Line segment merging function, can handle overlapping/partially overlapping segments,
    i.e., segments with similar slopes, close minimum distance, and endpoints far apart.

    Input parameters:
    lines: A list containing multiple lines, each represented by two coordinate points [x1,y1,x2,y2].
    angle_threshold: Angle threshold for similar line slopes (in degrees), default is 10.
    parallel_distance_threshold: Maximum perpendicular distance threshold between parallel line segments, default is 30.

    Returns:
    A list containing the merged lines.
    """
    if lines is None or len(lines) == 0:
        return []

    # Extract all line segments
    lines_array = lines
    merged_lines = []
    used_lines = [False] * len(lines_array)

    # Calculate line segment angle
    def get_angle(line):
        x1, y1, x2, y2 = line
        # Avoid division by zero error
        if x2 - x1 == 0:
            return 90.0
        return np.degrees(np.arctan((y2 - y1) / (x2 - x1)))

    # Calculate the distance between two line segments (minimum distance from point to line segment)
    def line_distance(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Direction vector of line segment 1
        v1 = np.array([x2 - x1, y2 - y1])
        # Unit vector of line segment 1
        len_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        if len_v1 == 0:
            return float('inf')
        unit_v1 = v1 / len_v1

        # Pad vectors to 3D for np.cross
        unit_v1_3d = np.array([unit_v1[0], unit_v1[1], 0])
        p3_vec_3d = np.array([x3 - x1, y3 - y1, 0])
        p4_vec_3d = np.array([x4 - x1, y4 - y1, 0])

        # Calculate the magnitude of the cross product (z-component)
        p1_to_line = np.abs(np.cross(unit_v1_3d, p3_vec_3d)[2])
        p2_to_line = np.abs(np.cross(unit_v1_3d, p4_vec_3d)[2])

        return min(p1_to_line, p2_to_line)

    # Determine if two line segments can be merged
    def can_merge(line1, line2):
        # Check angle difference
        angle1 = get_angle(line1)
        angle2 = get_angle(line2)
        angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
        if angle_diff > angle_threshold:
            return False

        # Check parallel distance
        distance = line_distance(line1, line2)
        if distance > parallel_distance_threshold:
            return False

        return True

    # Merge two line segments
    def merge_two_lines(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Sort all points by x-coordinate (or y-coordinate if close to vertical)
        angle = get_angle(line1)
        if abs(angle) > 45:
            # Vertical line, sort by y-coordinate
            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda p: p[1])
        else:
            # Horizontal line, sort by x-coordinate
            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda p: p[0])

        # Take the two farthest points as the endpoints of the new line segment
        return [points[0][0], points[0][1], points[3][0], points[3][1]]

    # Merging process
    for i in range(len(lines_array)):
        if used_lines[i]:
            continue

        current_line = lines_array[i]
        used_lines[i] = True

        # Try to merge other line segments
        while True:
            merged = False
            for j in range(len(lines_array)):
                if used_lines[j] or i == j:
                    continue

                if can_merge(current_line, lines_array[j]):
                    current_line = merge_two_lines(current_line, lines_array[j])
                    used_lines[j] = True
                    merged = True

            if not merged:
                break

        merged_lines.append(current_line)

    return merged_lines

# Identify horizontal truss members
def classify_H_truss_members(lines, threshold=20, angle_threshold=5):
    """
    Identify horizontal truss members
    Iterate through the input line segments. For each segment, calculate how many endpoints
    from other segments are within a perpendicular distance less than the threshold.
    Finally, select the two segments with the most adjacent endpoints as horizontal trusses.

    Input parameters:
    lines: List of line segments
    threshold: Distance threshold for adjacent endpoints
    angle_threshold: Maximum allowed angle between two horizontal trusses

    Returns:
    A dictionary containing "H-truss" and "Other" keys, representing horizontal trusses
    and other segments respectively.
    """
    if not lines or len(lines) < 3:
        return {"H-truss": [], "Other": lines.copy() if lines else []}

    # Calculate the perpendicular distance from a point to a line segment
    def perpendicular_distance(line, point):
        x1, y1, x2, y2 = line
        x0, y0 = point

        # If the segment length is close to zero, return the Euclidean distance from the point to the endpoint
        if abs(x2-x1) < 1e-8 and abs(y2-y1) < 1e-8:
            return np.sqrt((x0-x1)**2 + (y0-y1)**2)

        # Calculate segment length
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Calculate the area using the cross product, divide by length to get height (perpendicular distance)
        area = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))
        return area / length

    # Count the number of adjacent endpoints for each line segment
    adjacent_counts = []
    for i, line in enumerate(lines):
        count = 0
        for j, other_line in enumerate(lines):
            if i == j:
                continue

            # Get the endpoints of the other line segment
            endpoints = [(other_line[0], other_line[1]), (other_line[2], other_line[3])]

            # Check if the endpoint is near the current line segment
            for point in endpoints:
                dist = perpendicular_distance(line, point)
                if dist < threshold:
                    count += 1

        adjacent_counts.append((i, count, line))

    # Sort in descending order by the number of adjacent endpoints
    adjacent_counts.sort(key=lambda x: x[1], reverse=True)

    # Check if the top two segments are approximately parallel
    if len(adjacent_counts) >= 2:
        line1 = adjacent_counts[0][2]
        line2 = adjacent_counts[1][2]

        # Calculate the direction vector of the line segment
        vec1 = [line1[2] - line1[0], line1[3] - line1[1]]
        vec2 = [line2[2] - line2[0], line2[3] - line2[1]]

        # Calculate the length of the vector
        len1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
        len2 = np.sqrt(vec2[0]**2 + vec2[1]**2)

        # Normalize the vector
        if len1 > 0 and len2 > 0:
            vec1 = [vec1[0]/len1, vec1[1]/len1]
            vec2 = [vec2[0]/len2, vec2[1]/len2]

            # Calculate the dot product to determine if parallel (same or opposite direction)
            dot_product = abs(vec1[0]*vec2[0] + vec1[1]*vec2[1])

            # Calculate the angle between the two line segments (degrees)
            angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

            # If approximately parallel (dot product close to 1), check if the angle is within the threshold
            if dot_product > 0.9:  # Cosine value close to 1 indicates the angle is close to 0 or 180 degrees
                if angle <= angle_threshold:
                    # Calculate the slopes of the two lines
                    x1, y1, x2, y2 = line1
                    x3, y3, x4, y4 = line2

                    # Calculate the slopes of the two lines
                    slope1 = (y2 - y1) / (x2 - x1) if abs(x2 - x1) > 1e-8 else float('inf')
                    slope2 = (y4 - y3) / (x4 - x3) if abs(x4 - x3) > 1e-8 else float('inf')

                    # Determine if both line segments are nearly horizontal (slope close to 0)
                    slope_threshold = 0.1  # Define the threshold for slope close to 0
                    if abs(slope1) < slope_threshold and abs(slope2) < slope_threshold:
                        # Adjust the slope uniformly to 0 (completely horizontal)
                        avg_slope = 0
                        print("Detected parallel and nearly horizontal segments, adjusting slope to 0")
                    else:
                        # Handle the case of vertical lines
                        if slope1 == float('inf') and slope2 == float('inf'):
                            avg_slope = float('inf')
                        elif slope1 == float('inf'):
                            avg_slope = slope2
                        elif slope2 == float('inf'):
                            avg_slope = slope1
                        else:
                            avg_slope = (slope1 + slope2) / 2

                    # Adjust the line segment, keeping the midpoint unchanged
                    # Midpoint of segment 1
                    mid_x1 = (x1 + x2) / 2
                    mid_y1 = (y1 + y2) / 2
                    # Midpoint of segment 2
                    mid_x2 = (x3 + x4) / 2
                    mid_y2 = (y3 + y4) / 2

                    # Segment length
                    length1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    length2 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)

                    # Adjust endpoints of segment 1
                    if avg_slope == float('inf'):
                        # Vertical line
                        x1_new = mid_x1
                        y1_new = mid_y1 - length1/2
                        x2_new = mid_x1
                        y2_new = mid_y1 + length1/2
                    else:
                        # Calculate the x-offset for the new endpoints of segment 1
                        dx1 = length1 / (2 * np.sqrt(1 + avg_slope**2))
                        x1_new = mid_x1 - dx1
                        y1_new = mid_y1 - avg_slope * dx1
                        x2_new = mid_x1 + dx1
                        y2_new = mid_y1 + avg_slope * dx1

                    # Adjust endpoints of segment 2
                    if avg_slope == float('inf'):
                        # Vertical line
                        x3_new = mid_x2
                        y3_new = mid_y2 - length2/2
                        x4_new = mid_x2
                        y4_new = mid_y2 + length2/2
                    else:
                        # Calculate the x-offset for the new endpoints of segment 2
                        dx2 = length2 / (2 * np.sqrt(1 + avg_slope**2))
                        x3_new = mid_x2 - dx2
                        y3_new = mid_y2 - avg_slope * dx2
                        x4_new = mid_x2 + dx2
                        y4_new = mid_y2 + avg_slope * dx2

                    # Update line segments
                    line1 = [x1_new, y1_new, x2_new, y2_new]
                    line2 = [x3_new, y3_new, x4_new, y4_new]

                    # Build the list of other line segments
                    h_truss_indices = {adjacent_counts[0][0], adjacent_counts[1][0]}
                    other_lines = [lines[i] for i in range(len(lines)) if i not in h_truss_indices]

                    return {"H-truss": [line1, line2], "Other": other_lines}
                else:
                    print(f"Error: Detected horizontal truss angle is {angle:.2f} degrees, exceeding threshold {angle_threshold} degrees.")
                    # Build the list of other line segments
                    h_truss_indices = {adjacent_counts[0][0], adjacent_counts[1][0]}
                    other_lines = [lines[i] for i in range(len(lines)) if i not in h_truss_indices]

                    return {"H-truss": [line1, line2], "Other": other_lines}

    # If no suitable parallel segments are found, return the two segments with the most endpoints
    if len(adjacent_counts) >= 2:
        line1 = adjacent_counts[0][2]
        line2 = adjacent_counts[1][2]

        # Build the list of other line segments
        h_truss_indices = {adjacent_counts[0][0], adjacent_counts[1][0]}
        other_lines = [lines[i] for i in range(len(lines)) if i not in h_truss_indices]

        return {"H-truss": [line1, line2], "Other": other_lines}
    elif len(adjacent_counts) == 1:
        line1 = adjacent_counts[0][2]

        # Build the list of other line segments
        h_truss_indices = {adjacent_counts[0][0]}
        other_lines = [lines[i] for i in range(len(lines)) if i not in h_truss_indices]

        return {"H-truss": [line1], "Other": other_lines}
    else:
        return {"H-truss": [], "Other": []}

def classify_V_truss_members(lines, angle_threshold=20):
    """
    Identify vertical truss members and diagonal braces
    Iterate through the segments in 'lines'. If the angle between a segment and the H_truss_lines
    is less than angle_threshold, it's considered a vertical truss; otherwise, it's a diagonal brace.

    Input parameters:
    lines: List of line segments, including H-truss and Other
    angle_threshold: Angle threshold

    Returns:
    A dictionary with "H-truss", "V_truss", and "D_truss" keys, representing horizontal trusses,
    vertical trusses, and diagonal braces respectively.
    """
    h_truss_lines = lines.get("H-truss", [])
    other_lines = lines.get("Other", [])

    v_truss_lines = []
    d_truss_lines = []

    # If there are no horizontal trusses, vertical trusses cannot be determined
    if not h_truss_lines:
        return {"H-truss": h_truss_lines, "V_truss": [], "D_truss": other_lines}

    # Calculate the average angle of the horizontal trusses
    h_angles = []
    for line in h_truss_lines:
        x1, y1, x2, y2 = line
        # Avoid division by zero error
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
        h_angles.append(angle)

    h_avg_angle = sum(h_angles) / len(h_angles) if h_angles else 0

    # Iterate through other segments to determine if they are vertical trusses or diagonal braces
    for line in other_lines:
        x1, y1, x2, y2 = line
        # Avoid division by zero error
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))

        # Calculate the angle difference with the horizontal trusses
        # Vertical trusses should be at a 90-degree angle to horizontal trusses
        angle_diff = abs(abs(angle - h_avg_angle) - 90)

        # If the angle is close to 90 degrees (within the threshold), it's considered a vertical truss
        if angle_diff < angle_threshold:
            v_truss_lines.append(line)
        else:
            d_truss_lines.append(line)

    return {
        "H-truss": h_truss_lines,
        "V_truss": v_truss_lines,
        "D_truss": d_truss_lines
    }

# Add endpoint clustering and intersection alignment function
def cluster_endpoints(lines, threshold=6):
    """
    Cluster line segment endpoints and find intersection points to align segments better.
    Specific rules are as follows:
        1. First, merge the endpoints of vertical and diagonal truss members.
        2. Both endpoints of vertical and diagonal truss members should lie on the two horizontal truss lines respectively.
        3. Each endpoint of the horizontal truss members should coincide with at least one endpoint of a vertical/diagonal truss member.

    Parameters:
    lines: A dictionary with "H-truss", "V_truss", and "D_truss" keys, representing horizontal trusses,
           vertical trusses, and diagonal braces respectively.
    threshold: Distance threshold for endpoint clustering

    Returns:
    Updated dictionary of line segments
    """
    # Extract various types of line segments
    h_truss_lines = lines.get("H-truss", [])
    v_truss_lines = lines.get("V_truss", [])
    d_truss_lines = lines.get("D_truss", [])

    if not h_truss_lines:
        return lines  # If there are no horizontal trusses, alignment cannot be performed

    # Calculate the shortest distance from a point to a line segment and the closest point
    def point_to_line_distance(point, line):
        x0, y0 = point
        x1, y1, x2, y2 = line

        # If the segment length is 0, return the distance to the endpoint directly
        if abs(x2-x1) < 1e-8 and abs(y2-y1) < 1e-8:
            return np.sqrt((x0-x1)**2 + (y0-y1)**2), (x1, y1)

        # Direction vector of the line segment
        dx, dy = x2-x1, y2-y1
        # Square of the segment length
        length_squared = dx**2 + dy**2

        # Calculate the projection ratio t
        # Ensure t is within [0, 1] to stay within the segment
        t = max(0, min(1, ((x0-x1)*dx + (y0-y1)*dy) / length_squared))

        # Calculate the closest point on the line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Calculate the distance
        distance = np.sqrt((x0-closest_x)**2 + (y0-closest_y)**2)

        return distance, (closest_x, closest_y)

    # Collect endpoints of vertical and diagonal trusses
    vd_endpoints = []
    for i, line in enumerate(v_truss_lines):
        vd_endpoints.append(("V", i, 0, (line[0], line[1])))  # Start point
        vd_endpoints.append(("V", i, 1, (line[2], line[3])))  # End point

    for i, line in enumerate(d_truss_lines):
        vd_endpoints.append(("D", i, 0, (line[0], line[1])))  # Start point
        vd_endpoints.append(("D", i, 1, (line[2], line[3])))  # End point

    # Step 1: Merge and cluster endpoints of vertical and diagonal trusses
    clusters = []
    processed = set()

    for i, (type1, idx1, end1, point1) in enumerate(vd_endpoints):
        if i in processed:
            continue

        cluster = [(type1, idx1, end1, point1)]
        processed.add(i)

        for j, (type2, idx2, end2, point2) in enumerate(vd_endpoints):
            if j in processed or i == j:
                continue

            distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            if distance < threshold * 0.5:  # Use a smaller threshold for endpoint merging
                cluster.append((type2, idx2, end2, point2))
                processed.add(j)

        if len(cluster) > 1:  # Only focus on clusters with multiple points
            clusters.append(cluster)

    # Merge endpoint clusters
    for cluster in clusters:
        # Calculate the average position
        avg_x = sum(p[3][0] for p in cluster) / len(cluster)
        avg_y = sum(p[3][1] for p in cluster) / len(cluster)

        # Update the endpoints of the line segment each point belongs to
        for truss_type, line_idx, point_idx, _ in cluster:
            if truss_type == "V":
                if point_idx == 0:
                    v_truss_lines[line_idx][0] = avg_x
                    v_truss_lines[line_idx][1] = avg_y
                else:
                    v_truss_lines[line_idx][2] = avg_x
                    v_truss_lines[line_idx][3] = avg_y
            else:  # D
                if point_idx == 0:
                    d_truss_lines[line_idx][0] = avg_x
                    d_truss_lines[line_idx][1] = avg_y
                else:
                    d_truss_lines[line_idx][2] = avg_x
                    d_truss_lines[line_idx][3] = avg_y

    # Step 2: Ensure both endpoints of vertical and diagonal trusses snap to different horizontal trusses
    for v_idx, v_line in enumerate(v_truss_lines):
        # Process start and end points
        for point_idx in [0, 1]:
            x, y = v_line[point_idx*2], v_line[point_idx*2+1]

            # Find the nearest horizontal truss
            min_distance = float('inf')
            nearest_point = None
            nearest_h_idx = -1

            for h_idx, h_line in enumerate(h_truss_lines):
                distance, closest_point = point_to_line_distance((x, y), h_line)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = closest_point
                    nearest_h_idx = h_idx

            # If a sufficiently close horizontal truss is found, align the endpoint to it
            if min_distance < threshold and nearest_point:
                # Update vertical truss endpoint
                v_truss_lines[v_idx][point_idx*2] = nearest_point[0]
                v_truss_lines[v_idx][point_idx*2+1] = nearest_point[1]

    for d_idx, d_line in enumerate(d_truss_lines):
        # Process start and end points
        for point_idx in [0, 1]:
            x, y = d_line[point_idx*2], d_line[point_idx*2+1]

            # Find the nearest horizontal truss
            min_distance = float('inf')
            nearest_point = None
            nearest_h_idx = -1

            for h_idx, h_line in enumerate(h_truss_lines):
                distance, closest_point = point_to_line_distance((x, y), h_line)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = closest_point
                    nearest_h_idx = h_idx

            # If a sufficiently close horizontal truss is found, align the endpoint to it
            if min_distance < threshold and nearest_point:
                # Update diagonal truss endpoint
                d_truss_lines[d_idx][point_idx*2] = nearest_point[0]
                d_truss_lines[d_idx][point_idx*2+1] = nearest_point[1]

    # Step 3: Ensure horizontal truss endpoints align with the nearest vertical/diagonal truss endpoints
    # Goal: For each endpoint of a horizontal truss, if it's not sufficiently close (distance > threshold)
    #       to any vertical or diagonal truss endpoint, move it to the position of the nearest one.

    # Collect all updated endpoints of vertical and diagonal trusses
    updated_vd_points = []
    for line in v_truss_lines:
        updated_vd_points.append((line[0], line[1]))
        updated_vd_points.append((line[2], line[3]))
    for line in d_truss_lines:
        updated_vd_points.append((line[0], line[1]))
        updated_vd_points.append((line[2], line[3]))

    if not updated_vd_points:
        print("Warning: No vertical or diagonal truss endpoints found in Step 3. Cannot align horizontal truss endpoints.")
    else:
        # Check and adjust each horizontal truss endpoint
        for h_idx, h_line in enumerate(h_truss_lines):
            for point_idx in [0, 1]:  # Check start point (0) and end point (1)
                h_point_x, h_point_y = h_line[point_idx * 2], h_line[point_idx * 2 + 1]
                current_h_point = (h_point_x, h_point_y)

                # Find the nearest vertical/diagonal truss endpoint to the current horizontal truss endpoint
                min_distance_sq = float('inf')
                nearest_vd_point = None

                for vd_point in updated_vd_points:
                    # Use the square of the distance for comparison to avoid square root calculation
                    distance_sq = (current_h_point[0] - vd_point[0])**2 + (current_h_point[1] - vd_point[1])**2
                    if distance_sq < min_distance_sq:
                        min_distance_sq = distance_sq
                        nearest_vd_point = vd_point

                # Calculate the actual minimum distance
                min_distance = np.sqrt(min_distance_sq)

                # If the minimum distance is greater than the threshold, the horizontal truss endpoint needs alignment
                if min_distance > threshold:
                    # Move the horizontal truss endpoint to the position of the nearest vertical/diagonal truss endpoint
                    # (Ensure nearest_vd_point is not None, it can always be found if updated_vd_points is not empty)
                    if nearest_vd_point:
                        h_truss_lines[h_idx][point_idx * 2] = nearest_vd_point[0]
                        h_truss_lines[h_idx][point_idx * 2 + 1] = nearest_vd_point[1]
                        # Optional debug output:
                        # print(f"Adjusted H-truss {h_idx} endpoint {point_idx} from ({h_point_x:.1f}, {h_point_y:.1f}) to ({nearest_vd_point[0]:.1f}, {nearest_vd_point[1]:.1f}), distance: {min_distance:.1f}")

    # Step 4: Adjust the slope of vertical truss members to be perpendicular to horizontal members, rotating around the midpoint
    if h_truss_lines:
        # 计算水平桁架的平均斜率
        h_slopes = []
        for h_line in h_truss_lines:
            x1, y1, x2, y2 = h_line
            # 避免除零错误
            if abs(x2 - x1) < 1e-8:
                h_slopes.append(float('inf'))
            else:
                h_slopes.append((y2 - y1) / (x2 - x1))
        
        # 过滤掉无限值，计算平均斜率
        valid_h_slopes = [s for s in h_slopes if s != float('inf')]
        if valid_h_slopes:
            avg_h_slope = sum(valid_h_slopes) / len(valid_h_slopes)
        else:
            avg_h_slope = 0  # 如果所有斜率都是无限值，默认为0
        
        # 计算垂直于水平桁架的斜率
        if avg_h_slope == 0:
            perp_slope = float('inf')  # 垂直线
        elif abs(avg_h_slope) == float('inf'):
            perp_slope = 0  # 水平线
        else:
            perp_slope = -1 / avg_h_slope  # 垂直斜率
        
        # 调整每个垂直桁架的斜率
        for v_idx, v_line in enumerate(v_truss_lines):
            x1, y1, x2, y2 = v_line
            
            # 计算中点
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # 计算线段长度
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 根据垂直斜率调整端点
            if perp_slope == float('inf'):
                # 垂直线段
                new_x1 = mid_x
                new_y1 = mid_y - length / 2
                new_x2 = mid_x
                new_y2 = mid_y + length / 2
            else:
                # 计算线段新端点的x偏移量
                dx = length / (2 * np.sqrt(1 + perp_slope**2))
                new_x1 = mid_x - dx
                new_y1 = mid_y - perp_slope * dx
                new_x2 = mid_x + dx
                new_y2 = mid_y + perp_slope * dx
            
            # 更新垂直桁架的端点
            v_truss_lines[v_idx] = [new_x1, new_y1, new_x2, new_y2]
    
    # Step 5: Align diagonal brace endpoints to the nearest updated vertical truss endpoints
    if v_truss_lines:  # Execute only if vertical trusses exist
        # Collect all updated vertical truss endpoints
        updated_v_endpoints = []
        for v_line in v_truss_lines:
            updated_v_endpoints.append((v_line[0], v_line[1]))
            updated_v_endpoints.append((v_line[2], v_line[3]))

        if updated_v_endpoints: # Ensure there are endpoints to align with
            for d_idx, d_line in enumerate(d_truss_lines):
                for point_idx in [0, 1]: # Check start and end points
                    d_point_x, d_point_y = d_line[point_idx * 2], d_line[point_idx * 2 + 1]
                    current_d_point = (d_point_x, d_point_y)

                    # Find the nearest vertical truss endpoint to the current diagonal brace endpoint
                    min_distance_sq = float('inf')
                    nearest_v_point = None

                    for v_point in updated_v_endpoints:
                        distance_sq = (current_d_point[0] - v_point[0])**2 + (current_d_point[1] - v_point[1])**2
                        if distance_sq < min_distance_sq:
                            min_distance_sq = distance_sq
                            nearest_v_point = v_point

                    # Calculate the actual minimum distance
                    min_distance = np.sqrt(min_distance_sq)

                    # If the nearest vertical truss endpoint is close enough, move the diagonal brace endpoint to its position
                    if min_distance < threshold and nearest_v_point:
                        d_truss_lines[d_idx][point_idx * 2] = nearest_v_point[0]
                        d_truss_lines[d_idx][point_idx * 2 + 1] = nearest_v_point[1]
                        # Optional debug output
                        # print(f"Adjusted D-brace {d_idx} endpoint {point_idx} to V-endpoint ({nearest_v_point[0]:.1f}, {nearest_v_point[1]:.1f})")

    # Step 6: Update the endpoints of horizontal members to align with the updated vertical/diagonal truss endpoints
    if v_truss_lines or d_truss_lines: # Execute only if vertical or diagonal trusses exist
        # Collect updated endpoints from vertical trusses (already done in Step 5 if v_truss_lines exist)
        if 'updated_v_endpoints' not in locals():
             updated_v_endpoints = []
             for v_line in v_truss_lines:
                 updated_v_endpoints.append((v_line[0], v_line[1]))
                 updated_v_endpoints.append((v_line[2], v_line[3]))

        # Collect updated diagonal brace endpoints
        updated_d_endpoints = []
        for d_line in d_truss_lines:
            updated_d_endpoints.append((d_line[0], d_line[1]))
            updated_d_endpoints.append((d_line[2], d_line[3]))

        # Merge all possible connection points
        all_connection_points = updated_v_endpoints + updated_d_endpoints

        if all_connection_points:  # Ensure there are endpoints to align with
            for h_idx, h_line in enumerate(h_truss_lines):
                for point_idx in [0, 1]:  # Check the start and end points of the horizontal member
                    h_point_x, h_point_y = h_line[point_idx * 2], h_line[point_idx * 2 + 1]
                    current_h_point = (h_point_x, h_point_y)

                    # Find the nearest connection point (vertical or diagonal endpoint) to the current horizontal truss endpoint
                    min_distance_sq = float('inf')
                    nearest_connection_point = None

                    for connection_point in all_connection_points:
                        distance_sq = (current_h_point[0] - connection_point[0])**2 + (current_h_point[1] - connection_point[1])**2
                        if distance_sq < min_distance_sq:
                            min_distance_sq = distance_sq
                            nearest_connection_point = connection_point

                    # Calculate the actual minimum distance
                    min_distance = np.sqrt(min_distance_sq)

                    # If the nearest connection point is close enough, move the horizontal truss endpoint to its position
                    if min_distance < threshold and nearest_connection_point:
                        h_truss_lines[h_idx][point_idx * 2] = nearest_connection_point[0]
                        h_truss_lines[h_idx][point_idx * 2 + 1] = nearest_connection_point[1]
                        # Optional debug output
                        # print(f"Adjusted H-truss {h_idx} endpoint {point_idx} to connection point ({nearest_connection_point[0]:.1f}, {nearest_connection_point[1]:.1f})")

    # Build the updated line segment dictionary
    updated_lines = {
        "H-truss": h_truss_lines,
        "V_truss": v_truss_lines,
        "D_truss": d_truss_lines
    }

    return updated_lines

def normalize_truss_size(lines):
    """
    将桁架结构大小(长度)归一化，通过点的重新映射实现
    
    Args:
        lines: 包含 H-truss, V_truss, D_truss 的线段字典
        
    Returns:
        updated_lines: 归一化后的线段字典
        
    注意:
        x坐标: 按从小到大排序后映射到0,1,2,3...的整数序列
        y坐标: 保持二值映射(0表示下边界, 1表示上边界)
    """
    # 提取各类线段
    h_truss_lines = lines.get("H-truss", [])
    v_truss_lines = lines.get("V_truss", [])
    d_truss_lines = lines.get("D_truss", [])
     
    # 翻转所有线段的y坐标（1000-y）以处理cv2坐标系
    for line_list in [h_truss_lines, v_truss_lines, d_truss_lines]:
        for i, line in enumerate(line_list):
            line_list[i] = [
                line[0],  # x1
                1000 - line[1],  # y1
                line[2],  # x2
                1000 - line[3]   # y2
            ]
    
    # 1. 将所有端点收集到一个集合中，确保唯一性
    unique_points = set()
    
    for line in h_truss_lines + v_truss_lines + d_truss_lines:
        unique_points.add((line[0], line[1]))  # 起点
        unique_points.add((line[2], line[3]))  # 终点
    
    # 2. 构建点的字典，给每个点分配名称
    point_dict = {}
    for i, point in enumerate(unique_points):
        point_dict[f"P{i}"] = point
    
    # 3. 创建逆向查找映射（从坐标到点名称）
    point_lookup = {point: name for name, point in point_dict.items()}
    
    # 4. 构建线段字典
    lines_dict = {}
    
    # 添加水平桁架线段
    for i, line in enumerate(h_truss_lines):
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])
        lines_dict[f"H{i}"] = (point_lookup[p1], point_lookup[p2])
    
    # 添加垂直桁架线段
    for i, line in enumerate(v_truss_lines):
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])
        lines_dict[f"V{i}"] = (point_lookup[p1], point_lookup[p2])
    
    # 添加斜向桁架线段
    for i, line in enumerate(d_truss_lines):
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])
        lines_dict[f"D{i}"] = (point_lookup[p1], point_lookup[p2])
    
    # 5. X坐标归一化: 对所有点按x坐标排序，将它们映射到0,1,2,3...
    # 首先收集所有不同的 x 坐标
    all_x_coords = sorted(set(p[0] for p in unique_points))
    
    # 合并相近的 x 坐标 (差值小于10的视为同一点)
    merged_x_coords = []
    if all_x_coords:
        current_group = [all_x_coords[0]]
        
        for i in range(1, len(all_x_coords)):
            # 如果当前x与上一组最后一个x的差值小于10，则归入同一组
            if all_x_coords[i] - current_group[-1] < 10:
                current_group.append(all_x_coords[i])
            else:
                # 将当前组的平均值添加到结果中
                merged_x_coords.append(sum(current_group) / len(current_group))
                # 开始新的一组
                current_group = [all_x_coords[i]]
        
        # 添加最后一组的平均值
        merged_x_coords.append(sum(current_group) / len(current_group))
    
    # 为原始x坐标创建到合并后坐标的映射
    x_merge_map = {}
    for orig_x in all_x_coords:
        # 找到最接近的合并后坐标
        closest_merged_x = min(merged_x_coords, key=lambda merged_x: abs(merged_x - orig_x))
        x_merge_map[orig_x] = closest_merged_x
    
    # 创建从合并后的x坐标到索引的映射
    unique_x_coords = merged_x_coords
    x_map = {x: i for i, x in enumerate(unique_x_coords)}
    
    # 6. Y坐标归一化: 收集所有唯一的y坐标
    unique_y_coords = sorted(set(p[1] for p in unique_points))
    
    # 获取水平桁架的y坐标，用于确定上下边界
    h_truss_y_coords = []
    for line in h_truss_lines:
        h_truss_y_coords.append(line[1])  # y1
        h_truss_y_coords.append(line[3])  # y2
    
    # 7. 创建Y坐标的二值映射
    binary_y_map = {}
    if h_truss_y_coords:
        # 找出水平桁架的最小和最大y坐标作为下边界和上边界
        min_h_y = min(h_truss_y_coords)
        max_h_y = max(h_truss_y_coords)
        
        # 创建二值映射：0表示下边界，1表示上边界
        for y in unique_y_coords:
            # 计算与上下边界的距离
            dist_to_min = abs(y - min_h_y)
            dist_to_max = abs(y - max_h_y)
            
            # 将y坐标映射到最近的边界
            if dist_to_min <= dist_to_max:
                binary_y_map[y] = 0  # 下边界
            else:
                binary_y_map[y] = 1  # 上边界
    else:
        # 如果没有水平桁架，则使用最小/最大y坐标
        binary_y_map = {y: (1 if i > 0 else 0) for i, y in enumerate(unique_y_coords)}
    
    # 8. 创建坐标归一化映射
    normalized_coords = {}
    for point in unique_points:
        x, y = point
        # 使用合并后的x坐标值
        merged_x = x_merge_map[x]
        normalized_coords[point] = (x_map[merged_x], binary_y_map[y])
    
    # 9. 更新点字典中的坐标为归一化后的坐标
    normalized_point_dict = {}
    for name, coords in point_dict.items():
        normalized_point_dict[name] = normalized_coords[coords]
    
    # 10. 重建线段
    new_h_truss = []
    new_v_truss = []
    new_d_truss = []
    
    # 从线段字典中重建各类桁架
    for line_name, (p1_name, p2_name) in lines_dict.items():
        p1_coords = normalized_point_dict[p1_name]
        p2_coords = normalized_point_dict[p2_name]
        new_line = [p1_coords[0], p1_coords[1], p2_coords[0], p2_coords[1]]
        
        if line_name.startswith('H'):
            new_h_truss.append(new_line)
        elif line_name.startswith('V'):
            new_v_truss.append(new_line)
        elif line_name.startswith('D'):
            new_d_truss.append(new_line)
    
    # 11. 保持类型信息
    h_types = []
    v_types = []
    d_types = []
    
    for line_name, _ in sorted(lines_dict.items()):
        if line_name.startswith('H'):
            h_types.append(line_name)
        elif line_name.startswith('V'):
            v_types.append(line_name)
        elif line_name.startswith('D'):
            d_types.append(line_name)
    
    # 12. 构建更新后的线段字典
    updated_lines = {
        "H-truss": new_h_truss,
        "V_truss": new_v_truss,
        "D_truss": new_d_truss
    }
    
    return updated_lines

# Merge after detecting line segments
if lines is not None:
    # Fix: Convert lines to a standard Python list
    lines = [line[0] for line in lines]
    print(f"Original line segments: {len(lines)}")

    # Improved line segment merging logic
    lines = merge_lines(lines, angle_threshold=17, parallel_distance_threshold=40)
    print(f"Merged line segments: {len(lines)}")

    # Identify horizontal trusses
    truss_result = classify_H_truss_members(lines, threshold=25)
    h_truss_lines = truss_result["H-truss"]
    other_lines = truss_result["Other"]
    print(f"Identified horizontal trusses: {len(h_truss_lines)}")
    print(f"Other segments: {len(other_lines)}")

# Further classify vertical trusses and diagonal braces
all_truss_result = classify_V_truss_members(truss_result, angle_threshold=20)
h_truss_lines = all_truss_result["H-truss"]
v_truss_lines = all_truss_result["V_truss"]
d_truss_lines = all_truss_result["D_truss"]

print(f"Horizontal trusses: {len(h_truss_lines)}")
print(f"Vertical trusses: {len(v_truss_lines)}")
print(f"Diagonal braces: {len(d_truss_lines)}")

# Perform endpoint clustering
clustered_truss_result = cluster_endpoints(all_truss_result, threshold=50)

# Normalize truss size
normalized_truss_result = normalize_truss_size(clustered_truss_result)

h_clustered_lines = normalized_truss_result["H-truss"]
v_clustered_lines = normalized_truss_result["V_truss"]
d_clustered_lines = normalized_truss_result["D_truss"]

print(h_clustered_lines)
print(v_clustered_lines)
print(d_clustered_lines)
