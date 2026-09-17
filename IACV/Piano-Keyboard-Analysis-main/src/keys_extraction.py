from copy import deepcopy

import cv2
import skimage.io
import numpy as np
import os
from matplotlib import image, pyplot as plt
import skimage.io
import json
import math
import inspect
from skimage.feature import match_template
from sklearn.cluster import DBSCAN

from logger import logger

import os
import cv2
import numpy as np

def _save_step(name, img):
    os.makedirs("outputs/key_steps", exist_ok=True)
    if img is None:
        return

    # uint8 safety
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # grayscale
    if len(img.shape) == 2:
        cv2.imwrite(f"outputs/key_steps/{name}.png", img)
        return

    # RGB/RGBA -> BGR for OpenCV saving
    if img.shape[2] == 4:
        out = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"outputs/key_steps/{name}.png", out)

import json
import os

def _save_roi_json(x, y, w, h, out_path="outputs/piano_roi.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    roi = {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h)
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(roi, f, indent=2)


class Key:
    def __init__(self, key_img, y_ul, x_ul, y_dr, x_dr, name):
        self.y_ul = y_ul
        self.x_ul = x_ul
        self.y_dr = y_dr
        self.x_dr = x_dr

        self.key_img = key_img
        self.name = name

    def coords(self):
        """

        :return: y_ul, x_ul, y_dr, x_dr
        """
        return self.y_ul, self.x_ul, self.y_dr, self.x_dr

    def image(self):
        return self.key_img

    def __str__(self):
        return self.name


class WhiteKey(Key):
    def __init__(self, key_img, y_ul, x_ul, y_dr, x_dr, name):
        super().__init__(key_img, y_ul, x_ul, y_dr, x_dr, name)
        if self.name[0] in "CDFGA":
            self.black_is_next = True
        else:
            self.black_is_next = False


class BlackKey(Key):
    def __init__(self, key_img, y_ul, x_ul, y_dr, x_dr, name):
        super().__init__(key_img, y_ul, x_ul, y_dr, x_dr, name)


def _key_to_dict(k: Key):
    y1, x1, y2, x2 = k.coords()
    return {"y_ul": int(y1), "x_ul": int(x1), "y_dr": int(y2), "x_dr": int(x2), "name": str(k.name)}

def _save_keys_json(keys_dict, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {name: _key_to_dict(k) for name, k in keys_dict.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _load_keys_json(json_path, key_cls, key_img):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for name, v in data.items():
        out[name] = key_cls(
            key_img,
            int(v["y_ul"]), int(v["x_ul"]),
            int(v["y_dr"]), int(v["x_dr"]),
            str(v.get("name", name))
        )
    return out


key_names = {
    0: "C",
    1: "D",
    2: "E",
    3: "F",
    4: "G",
    5: "A",
    6: "B",
}


class KeysExtractorThroughLines:
    def __init__(self):
        self.logger = logger

        ref_piano_path = os.path.join("data", "octave", "octava.png")
        ref_piano = skimage.io.imread(ref_piano_path)
        self.ref_piano = cv2.cvtColor(ref_piano, cv2.COLOR_RGB2GRAY)
        self.logger.info("Keys Extractor created")
        self.debug = True
        self.cache_white_json = "data/octave/octava_w.json"
        self.cache_black_json = "data/octave/octava_b.json"
        self.cache_enabled = True 


    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _find_piano_contour(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = cv2.adaptiveThreshold(image_gray, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)

        return image_gray, c

    def _draw_contour(self, image, c):
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return x, y, w, h

    def _create_piano_masked(self, image, c):
        mask = np.zeros_like(image)
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        image[mask == 0] = 0

        return image

    def _find_contour_angle(self, image, c):
        center, hw, angle = cv2.minAreaRect(c)
        h, w = hw

        if w > h:
            angle = 90 + angle

        self.logger.debug(f"Found h of contour is {np.round(h, 2)} px, width is {np.round(w, 2)} px")
        self.logger.debug(f"Found angle of contour is {np.round(angle, 2)} degrees")

        return angle
        # box = cv2.boxPoints(rect)

    def _find_kp(self, image):
        sift = cv2.SIFT.create()
        kp, des = sift.detectAndCompute(image, None)
        return kp, des

    def build_homography(self, img1, kp1, des1, img2, kp2, des2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)

        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)],
                                 True, 255, 3, cv2.LINE_AA)

        else:
            self.logger.debug("Not enough matches are found - {}/{}".format(len(good), 10))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        plt.imshow(img3, 'gray'), plt.show()

    def _calculate_similarity(self, frame1, frame2):
        h1, w1 = frame1.shape
        h2, w2 = frame2.shape

        h_min = min(h1, h2)
        w_min = min(w1, w2)

        frame1 = deepcopy(frame1[:h_min, :w_min, ...])
        frame2 = deepcopy(frame2[:h_min, :w_min, ...])

        frame1[frame1 > 127] = 255
        frame1[frame1 <= 127] = 0

        frame2[frame2 > 127] = 255
        frame2[frame2 <= 127] = 0

        black1 = np.sum(frame1 == 0)
        black2 = np.sum(frame2 == 0)

        white1 = np.sum(frame1 == 255)
        white2 = np.sum(frame2 == 255)

        eps = 1e-9
        score = (black1 / (white1 + eps)) / (black2 / (white2 + eps) + eps)

        if score > 1:
            score = 1 / score

        return score
    
    
    def _make_white_key_obj(self, key_img, x1, y1, x2, y2, name):
        # WhiteKey(key_img, y_ul, x_ul, y_dr, x_dr, name)
        return WhiteKey(
            key_img,
            int(y1), int(x1),
            int(y2), int(x2),
            str(name)
        )


    def _build_white_keys_from_boundaries(self, roi_gray, x_off, y_off, yk0, yk1, boundaries, n_white=52):
        h, w = roi_gray.shape
        y1 = max(0, int(yk0))
        y2 = min(h - 1, int(yk1))

        b = np.clip(np.asarray(boundaries, dtype=np.int32), 0, w - 1)
        b = np.unique(np.sort(b))

        if len(b) >= 2:
            diffs = np.diff(b).astype(np.float32)
            diffs = diffs[(diffs > 4) & (diffs < 200)]
            if len(diffs) > 0:
                step = float(np.median(diffs))
            else:
                step = w / float(n_white)
        else:
            step = w / float(n_white)

        out = []
        if len(b) > 0 and b[0] > 0.6 * step:
            out.append(0)
        elif len(b) > 0:
            out.append(int(b[0]))
        else:
            out.append(0)

        cur = out[0]
        for _ in range(n_white):
            target = cur + step
            cand = b[(b > cur + 0.3 * step) & (b < cur + 1.7 * step)]
            if len(cand) > 0:
                nxt = int(cand[np.argmin(np.abs(cand - target))])
            else:
                nxt = int(round(target))
            out.append(nxt)
            cur = nxt

        b = np.clip(np.array(out, dtype=np.int32), 0, w - 1)
        for i in range(1, len(b)):
            if b[i] <= b[i - 1]:
                b[i] = min(w - 1, b[i - 1] + 1)

        white_keys = {}
        for i in range(len(b) - 1):
            x1, x2 = int(b[i]), int(b[i + 1])
            if x2 <= x1 + 2:
                continue
            name = f"W_{i}"
            white_keys[name] = WhiteKey(
                roi_gray[y1:y2, x1:x2],
                int(y_off + y1), int(x_off + x1),
                int(y_off + y2), int(x_off + x2),
                name
            )

        return white_keys


    def _find_white_keys_coords(self, masked_piano, x, y, w, h, white_key_w):
        masked_piano = masked_piano.astype(np.int8)

        keys_dict = {}
        patch = masked_piano[y:y + h, x:x + w, ...]

        # --- HARD FIX for adaptiveThreshold input ---
        print("PATCH BEFORE:", patch.shape, patch.dtype)

        # if patch is RGBA/RGB/BGR, convert to gray
        if len(patch.shape) == 3:
            if patch.shape[2] == 4:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2GRAY)
            else:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # force uint8
        if patch.dtype != np.uint8:
            # if patch is float in [0,1], scale to [0,255]
            if patch.dtype in [np.float32, np.float64] and patch.max() <= 1.0:
                patch = (patch * 255.0).astype(np.uint8)
            else:
                patch = np.clip(patch, 0, 255).astype(np.uint8)

        # make it contiguous (OpenCV sometimes fails without this)
        patch = np.ascontiguousarray(patch)

        print("PATCH AFTER:", patch.shape, patch.dtype)

        patch = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 199, 5)
        
        # ensure patch is grayscale uint8 for adaptiveThreshold
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)  # or BGR2GRAY if patch is BGR
        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)

        ref_piano_h, ref_piano_w = self.ref_piano.shape[:2]
        ref_piano_new_w = int(white_key_w * 7)
        ref_piano_new_h = int(ref_piano_h * (ref_piano_new_w / ref_piano_w))
        ref_piano_resh = cv2.resize(self.ref_piano, (ref_piano_new_w, ref_piano_new_h))

        result = match_template(patch, ref_piano_resh)


        res_max = np.max(result)

        mask = np.zeros_like(result)
        mask[result > res_max * 0.85] = 1


        peaks_coords_y, peaks_coords_x = np.where(mask == 1)

        peaks_coords = np.array(list(zip(peaks_coords_y, peaks_coords_x)))
        clustering_octave = DBSCAN(eps=3, min_samples=2).fit(peaks_coords)
        labels_octave = clustering_octave.labels_

        # left upper coordinates of the "do(C)" key in each found octave
        octave_coords_lu = []
        for label in np.unique(labels_octave):
            coord_y = int(np.mean(peaks_coords_y[labels_octave == label])) + y
            coord_x = int(np.mean(peaks_coords_x[labels_octave == label])) + x

            for j in range(5):
                if np.min(masked_piano[coord_y, coord_x - 5:coord_x + 5]) - np.min(
                        masked_piano[coord_y - j, coord_x - 5:coord_x + 5]) > 150:
                    coord_y = coord_y - j
                    break
            octave_coords_lu.append([coord_y, coord_x])

        octave_coords_lu.sort(key=lambda x: x[1])
        diff = [octave_coords_lu[i + 1][1] - octave_coords_lu[i][1] for i in range(len(octave_coords_lu) - 1)]
        diff.append(diff[-1])

        octave_coords_lu = np.array(octave_coords_lu)
        img_h, img_w = masked_piano.shape[:2]

        y_coord_bottom_keys = []
        for coord in octave_coords_lu:
            y, x = coord
            for j in range(y, img_h):
                if np.mean(masked_piano[j:j + 3, x - 5:x + 5]) / np.mean(masked_piano[j + 3:j + 6, x - 5:x + 5]) > 2:
                    y_coord_bottom_keys.append(j + 3)
                    break
        h_keys = int(np.median(y_coord_bottom_keys) - np.median(octave_coords_lu[:, 0]))

        self.logger.info(f"Found height of white keys is {h_keys}")

        all_white_keys_coords = []
        num_octave = 1
        # add coordinates of keys in found octaves
        for idx, coord in enumerate(octave_coords_lu):
            y, x = coord
            w_keys = diff[idx] / 7

            for i in range(7):
                name_cur = f"{key_names[i]}_{num_octave}"
                key_cur = WhiteKey(masked_piano[y: y + h_keys, int(x + i * w_keys):int(x + (i + 1) * w_keys)],
                                   y, int(x + i * w_keys), y + h_keys, int(x + (i + 1) * w_keys), name_cur)
                keys_dict[name_cur] = key_cur
                all_white_keys_coords.append([y, int(x + i * w_keys), y + h_keys, int(x + (i + 1) * w_keys)])

            num_octave += 1

        # add coordinates of keys not in full octave,
        # going before the first octave
        start_y, start_x, _, _ = keys_dict["C_1"].coords()
        for j in range(14):
            if int(start_x - white_key_w * (j % 7 + 1)) + 5 < 0:
                break

            patch_key = masked_piano[start_y: start_y + h_keys - 1,
                        int(start_x - white_key_w * (j % 7 + 1)): int(start_x - white_key_w * (j % 7))]
            name_key = key_names[6 - j % 7]
            true_key = keys_dict[f"{name_key}_2"].image()

            score = self._calculate_similarity(patch_key, true_key)

            if score < 0.5:
                break

            name_cur = f"{name_key}_0"
            key_cur = WhiteKey(patch_key, start_y, int(start_x - white_key_w * (j % 7 + 1)),
                               start_y + h_keys - 1,
                               int(start_x - white_key_w * (j % 7)),
                               name_cur)
            keys_dict[name_cur] = key_cur

        # add coordinates of keys not in full octave,
        # going after the last octave
        start_y, _, _, start_x = keys_dict[f"B_{num_octave - 1}"].coords()
        for j in range(14):
            if int(start_x + white_key_w * (j % 7 + 1)) - 5 > img_w:
                break

            patch_key = masked_piano[start_y: start_y + h_keys - 1,
                        int(start_x + white_key_w * (j % 7)): int(start_x + white_key_w * ((j % 7) + 1))]

            name_key = key_names[j]
            true_key = keys_dict[f"{name_key}_2"].image()

            score = self._calculate_similarity(patch_key, true_key)
            if score < 0.5:
                # it can be the last "C" which looks different from usual "C"
                if np.sum(patch_key > 200) / (patch_key.shape[0] * patch_key.shape[1]) < 0.8:
                    break

            name_cur = f"{name_key}_{num_octave}"
            key_cur = WhiteKey(patch_key, start_y, int(start_x + white_key_w * (j % 7)),
                               start_y + h_keys - 1,
                               int(start_x + white_key_w * (j % 7 + 1)),
                               name_cur)
            keys_dict[name_cur] = key_cur

        return keys_dict, h_keys

    def _keyboard_band_from_vertical_edges(self, roi_gray):
        gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
        gx = np.abs(gx)

        row_energy = gx.mean(axis=1)
        row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 31), 0).ravel()

        thr = 0.35 * row_energy.max()
        idx = np.where(row_energy > thr)[0]
        if len(idx) < 10:
            h = roi_gray.shape[0]
            return int(0.25 * h), int(0.80 * h)

        starts = [idx[0]]
        ends = []
        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1] + 1:
                ends.append(idx[i - 1])
                starts.append(idx[i])
        ends.append(idx[-1])

        lengths = [e - s for s, e in zip(starts, ends)]
        k = int(np.argmax(lengths))
        y0, y1 = int(starts[k]), int(ends[k])

        pad = int(0.05 * (y1 - y0 + 1))
        y0 = max(0, y0 - pad)
        y1 = min(roi_gray.shape[0] - 1, y1 + pad)
        return y0, y1

    def _split_wide_blob(self, bw_crop, expected_w, min_w, max_w):
        col = bw_crop.mean(axis=0).astype(np.float32)
        col = cv2.GaussianBlur(col.reshape(1, -1), (1, 21), 0).ravel()

        thr = 0.35 * col.max()
        m = col > thr

        runs = []
        i = 0
        n = len(m)
        while i < n:
            if not m[i]:
                i += 1
                continue
            j = i + 1
            while j < n and m[j]:
                j += 1
            runs.append((i, j))
            i = j

        segs = [(a, b) for (a, b) in runs if (b - a) >= 0.6 * min_w]

        if len(segs) <= 1:
            w = bw_crop.shape[1]
            k = max(1, int(round(w / (expected_w + 1e-9))))
            if k == 1:
                return [(0, w)]
            step = w / float(k)
            out = []
            for t in range(k):
                a = int(round(t * step))
                b = int(round((t + 1) * step))
                if b - a >= 0.6 * min_w:
                    out.append((a, b))
            return out

        out = []
        for a, b in segs:
            w = b - a
            if w <= 1.5 * max_w:
                out.append((a, b))
            else:
                k = max(1, int(round(w / (expected_w + 1e-9))))
                step = w / float(k)
                for t in range(k):
                    aa = int(round(a + t * step))
                    bb = int(round(a + (t + 1) * step))
                    if bb - aa >= 0.6 * min_w:
                        out.append((aa, bb))
        return out

    def _local_percentile_bw(self, img_u8, pct=40, chunk=None):
        h, w = img_u8.shape
        if chunk is None:
            chunk = max(120, w // 10)

        bw = np.zeros_like(img_u8, dtype=np.uint8)
        for a in range(0, w, chunk):
            b = min(w, a + chunk)
            t = np.percentile(img_u8[:, a:b], pct)
            bw[:, a:b] = (img_u8[:, a:b] < t).astype(np.uint8) * 255
        return bw

    def _tighten_black_bbox(self, gray_strip, x, y, w, h, pct=45, col_occ=0.25):
        patch = gray_strip[y:y + h, x:x + w]
        if patch.size == 0:
            return x, w

        t = np.percentile(patch, pct)
        m = (patch < t).astype(np.uint8)

        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        col = m.mean(axis=0)
        cols = np.where(col > col_occ)[0]
        if cols.size < 3:
            return x, w

        left = int(cols[0])
        right = int(cols[-1])

        left = max(0, left - 1)
        right = min(w - 1, right + 1)

        x_new = x + left
        w_new = (right - left + 1)
        return x_new, w_new

    def _find_black_keys_by_contours(self, roi_gray, x_off, y_off, yk0, yk1, white_key_w):
        key_h = yk1 - yk0 + 1

        y_strip0 = yk0 + int(0.05 * key_h)
        y_strip1 = yk0 + int(0.65 * key_h)
        strip = roi_gray[y_strip0:y_strip1, :]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        strip_eq = clahe.apply(strip)

        bw = self._local_percentile_bw(strip_eq, pct=40, chunk=max(120, strip_eq.shape[1] // 10))

        hline = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, 41), np.uint8), iterations=1)
        bw = cv2.subtract(bw, hline)

        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((11, 3), np.uint8), iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        _save_step("06b_black_bw", bw)

        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        black_keys = {}
        idx = 0

        expected_w = 0.60 * white_key_w
        min_w = 0.28 * white_key_w
        max_w = 1.30 * white_key_w

        min_h = 0.25 * key_h
        max_h = 0.95 * key_h

        raw_widths = []
        tmp_rects = []
        for c in cnts:
            x, y, ww, hh = cv2.boundingRect(c)

            if hh < min_h or hh > max_h:
                continue
            if ww < min_w:
                continue

            aspect = hh / (ww + 1e-9)
            if aspect < 1.2:
                continue

            raw_widths.append(ww)
            tmp_rects.append((x, y, ww, hh))

        w_med = float(np.median(raw_widths)) if len(raw_widths) else expected_w

        for x, y, ww, hh in tmp_rects:
            if ww <= max_w:
                segs = [(0, ww)]
            else:
                bw_crop = bw[y:y + hh, x:x + ww]
                segs = self._split_wide_blob(bw_crop, expected_w, min_w, max_w)

            for a, b in segs:
                xx = x + a
                wseg = b - a
                if wseg < min_w or wseg > 2.0 * max_w:
                    continue

                if wseg > 1.30 * w_med:
                    xx, wseg = self._tighten_black_bbox(strip_eq, xx, y, wseg, hh)

                name = f"BK_{idx}"
                x_abs1 = x_off + xx
                x_abs2 = x_off + xx + wseg
                y_abs1 = y_off + y_strip0 + y
                y_abs2 = y_off + y_strip0 + y + hh

                black_keys[name] = BlackKey(
                    roi_gray[(y_strip0 + y):(y_strip0 + y + hh), xx:(xx + wseg)],
                    y_abs1, x_abs1, y_abs2, x_abs2, name
                )
                idx += 1

        return dict(sorted(black_keys.items(), key=lambda kv: kv[1].coords()[1]))

    def _find_white_keys_w(self, image, yk0, yk1, n_white=52):
        """
        image: grayscale ROI (uint8)
        returns: boundaries (x positions), length n_white+1
        Robust: grid-first (w/52), peaks only for snapping.
        """

        # 1) smooth + boost contrast
        gray = cv2.GaussianBlur(image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 2) vertical edges only (Sobel x)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gx = np.abs(gx)

        # 3) use only the key area (lower part of the keyboard band)
        h, w = gx.shape
        band_h = (yk1 - yk0 + 1)

        y1 = yk0 + int(0.65 * band_h)
        y2 = yk0 + int(0.98 * band_h)
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        gx_roi = gx[y1:y2, :]

        # 4) vertical projection
        proj = gx_roi.sum(axis=0)
        proj = proj / (proj.max() + 1e-9)
        proj = cv2.GaussianBlur(proj.reshape(1, -1), (1, 31), 0).ravel()

        # ---- expected grid step ----
        step0 = w / float(n_white)
        min_dist = int(max(3, 0.5 * step0))
        snap_r = 0.35 * step0

        # 5) collect peak candidates (simple local maxima)
        thr = 0.20 * proj.max()
        cand = []
        for x in range(2, w - 2):
            if proj[x] > thr and proj[x] >= proj[x - 1] and proj[x] >= proj[x + 1]:
                cand.append(x)
        cand = np.array(cand, dtype=np.int32)

        # 6) non-maximum suppression so peaks aren't too dense
        if cand.size > 0:
            order = cand[np.argsort(proj[cand])[::-1]]
            kept = []
            for x in order:
                if all(abs(x - k) >= min_dist for k in kept):
                    kept.append(int(x))
            kept = np.array(sorted(kept), dtype=np.int32)
        else:
            kept = np.array([], dtype=np.int32)

        # 7) choose best phase offset for the grid (maximize proj on boundaries)
        offsets = np.linspace(0.0, max(1.0, step0 - 1.0), 25)
        best_off, best_score = 0.0, -1.0
        for off in offsets:
            xs = off + np.arange(0, n_white + 1) * step0
            xi = np.clip(xs.astype(np.int32), 0, w - 1)
            score = float(proj[xi].sum())
            if score > best_score:
                best_score, best_off = score, off

        grid = best_off + np.arange(0, n_white + 1) * step0

        # 8) snap each grid boundary to nearest strong peak (if any)
        out = []
        for g in grid:
            if kept.size > 0:
                near = kept[np.abs(kept - g) <= snap_r]
            else:
                near = np.array([], dtype=np.int32)

            if near.size > 0:
                x = int(near[np.argmax(proj[near])])
            else:
                x = int(round(g))
            out.append(x)

        out = np.array(out, dtype=np.int32)
        out[0] = 0
        out[-1] = w - 1
        out = np.clip(out, 0, w - 1)

        # enforce strictly increasing
        for i in range(1, len(out)):
            if out[i] <= out[i - 1]:
                out[i] = min(w - 1, out[i - 1] + 1)

        return out


    def _find_orientation(self, image, y, h):
        mask = cv2.adaptiveThreshold(image, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
        _save_step("04_white_width_mask", mask)

        rows_max = np.mean(mask, axis=1)
        rows_piano = rows_max[y:y + h]
        first_part_mean = np.mean(rows_piano[:len(rows_piano) // 2])
        second_part_mean = np.mean(rows_piano[len(rows_piano) // 2:])

        if second_part_mean < first_part_mean:
            return False

        return True

    def __call__(self, image: np.ndarray):
        """
        this method will return coordinates of white
        and black keys in the current image
        (the image should be without hands!) """

        if self.debug:
            # simulate rotated image
            image = self.rotate_image(image, 0)

        to_draw_img = deepcopy(image)
        _save_step("01_full_input", image)

        # find the initial piano contour
        image_gray, c = self._find_piano_contour(image)

        # define the rotation angle of the contour
        angle = self._find_contour_angle(image_gray, c)

        # rotate image on this angle
        image = self.rotate_image(image, angle)

        to_draw_img = self.rotate_image(to_draw_img, angle)

        # find the new (angle is 0) piano contour
        image_gray, c = self._find_piano_contour(image)

        # draw piano contour on image
        x, y, w, h = self._draw_contour(image, c)
        _save_step("02_roi_box", image)

        # delete irrelevant info from image
        masked_piano = self._create_piano_masked(image_gray, c)
        _save_step("03_masked_piano", masked_piano)

        # if the piano is upside-down, it should be rotated by 180 degrees
        is_correct_orientation = self._find_orientation(masked_piano, y, h)

        if not is_correct_orientation:
            masked_piano = self.rotate_image(masked_piano, 180)
            to_draw_img = self.rotate_image(to_draw_img, 180)
            angle += 180

        # build a tight ROI from masked_piano to kill black padding
        ys, xs = np.where(masked_piano > 5)
        if len(ys) == 0 or len(xs) == 0:
            raise ValueError("No non-zero pixels found in masked piano ROI.")

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        roi_gray = masked_piano[y0:y1 + 1, x0:x1 + 1]
        roi_rgb = to_draw_img[y0:y1 + 1, x0:x1 + 1]
        _save_step("03b_tight_roi", roi_rgb)

        yk0, yk1 = self._keyboard_band_from_vertical_edges(roi_gray)

        dbg = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(dbg, (0, yk0), (roi_gray.shape[1] - 1, yk1), (0, 255, 0), 2)
        _save_step("03c_keyboard_band", dbg)

        boundaries = self._find_white_keys_w(roi_gray, yk0, yk1)

        white_keys_coords = self._build_white_keys_from_boundaries(
            roi_gray, x0, y0, yk0, yk1, boundaries, n_white=52
        )

        # masked_piano is grayscale ROI with background removed
        # to_draw_img is rotated full image used for overlays

        if self.cache_enabled and os.path.exists(self.cache_white_json) and os.path.exists(self.cache_black_json):
            # Cached mode: JSON is loaded and drawn on full image
            white_keys_coords = _load_keys_json(self.cache_white_json, WhiteKey, masked_piano)
            black_keys_coords = _load_keys_json(self.cache_black_json, BlackKey, masked_piano)
            angle_out = angle
        else:
            # Normal mode: detection is performed
            masked_piano, white_key_w = self._find_white_keys_w(masked_piano)
            white_key_h = int(0.95 * h)

            white_keys_coords = self._build_white_keys_grid(x, y, w, h, white_key_w, white_key_h, masked_piano)
            black_key_w = self._find_black_keys_w(masked_piano, x, y, w, h, white_key_h, white_keys_coords)
            black_keys_coords = self._find_black_keys_coords(masked_piano, black_key_w, white_keys_coords)

            # Save JSON outputs from detector
            _save_keys_json(white_keys_coords, self.cache_white_json)
            _save_keys_json(black_keys_coords, self.cache_black_json)
            angle_out = angle

        # ---- overlays (always produced) ----
        white_vis = deepcopy(to_draw_img)
        if len(white_vis.shape) == 2:
            white_vis = cv2.cvtColor(white_vis, cv2.COLOR_GRAY2RGB)
        for k in white_keys_coords.values():
            y1, x1, y2, x2 = k.coords()
            cv2.rectangle(white_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _save_step("05_white_keys_only", white_vis)


        # find h and w of black keys
        black_key_w = self._find_black_keys_w(masked_piano, x, y, w, h, white_key_h, white_keys_coords)
        black_keys_coords = self._find_black_keys_coords(masked_piano, black_key_w, white_keys_coords)
        
        # black-only overlay
        black_vis = deepcopy(to_draw_img)
        if len(black_vis.shape) == 2:
            black_vis = cv2.cvtColor(black_vis, cv2.COLOR_GRAY2RGB)
        for k in black_keys_coords.values():
            y1, x1, y2, x2 = k.coords()
            cv2.rectangle(black_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        _save_step("06_black_keys_only", black_vis)

        final_vis = deepcopy(white_vis)
        for k in black_keys_coords.values():
            y1, x1, y2, x2 = k.coords()
            cv2.rectangle(final_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        _save_step("07_final_white_black", final_vis)

        x, y, w, h = self._draw_contour(image, c)
        _save_roi_json(x, y, w, h)


        return white_keys_coords, black_keys_coords, angle_out



def main():
    """
    test KeysExtractorThroughLines and other extractors here
    :return:
    """
    keys_extractor = KeysExtractorThroughLines()
    img_path = os.path.join("data", "frames", "piano_video_1", "cleaned", "cleaned_frame_2.png")
    image = skimage.io.imread(img_path)

    keys_extractor(image)


if __name__ == "__main__":
    main()