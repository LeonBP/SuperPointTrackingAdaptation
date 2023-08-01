# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

# This script is based on an original implementation by True Price.

import sys
import sqlite3
import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if blob is None:
        return np.array([])
    elif IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def get_table_names(self):
        cursor = self.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]

    def delete_two_view_geometry(self):
        self.execute("DELETE FROM two_view_geometries")

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def get_all_cameras(self):
        cursor = self.execute("SELECT * FROM cameras")
        return cursor.fetchall()

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def get_image_id(self, image_id):
        cursor = self.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))
        return cursor.fetchall()

    def get_image_id_from_name(self, name):
        id = self.execute("SELECT image_id FROM images WHERE name = ?", (name,))
        return id.fetchall()

    def get_all_images(self):
        cursor = self.execute("SELECT * FROM images")
        return cursor.fetchall()

    def get_all_image_names(self):
        cursor = self.execute("SELECT name FROM images")
        return cursor.fetchall()

    def get_all_image_ids(self):
        cursor = self.execute("SELECT image_id FROM images")
        return cursor.fetchall()

    def update_image_name(self, image_id, name):
        self.execute("UPDATE images SET name = ? WHERE image_id = ?", (name, image_id))

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def update_keypoints(self, id, kPS):
        assert(len(kPS.shape) == 2)
        assert(kPS.shape[1] in [2, 4, 6])

        kPS = np.asarray(kPS, np.float32)

        self.execute(
            '''UPDATE keypoints
               SET data=?,
               rows=?,
               cols=?   
               WHERE image_id=?''',
            (array_to_blob(kPS),)+kPS.shape+(id,)

                )

    def get_keypoints(self, id):
        rows = self.execute(
            '''SELECT * from keypoints  
            WHERE image_id=?''',
            (id,))
        return rows.fetchall()

    def get_all_keypoints(self):
        rows = self.execute(
            '''SELECT * from keypoints''')
        return rows.fetchall()

    def get_num_keypoints_id(self, id):
        rows = self.execute(
            '''SELECT rows from keypoints  
            WHERE image_id=?''',
            (id,))
        return rows.fetchall()

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def update_descriptors(self, id, descs):
        assert(len(descs.shape) == 2)
        assert(descs.shape[1] in [128,256])

        descs = np.ascontiguousarray(descs, np.uint8)

        self.execute(
            '''UPDATE descriptors
               SET data=?,
               rows=?,
               cols=?   
               WHERE image_id=?''',
            (array_to_blob(descs),)+descs.shape+(id,)

                )

    def get_descriptors(self, id):
        rows = self.execute(
            '''SELECT * from descriptors  
            WHERE image_id=?''',
            (id,))
        return rows.fetchall()

    def get_all_descriptors(self):
        rows = self.execute(
            '''SELECT * from descriptors''')
        return rows.fetchall()

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def update_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            '''UPDATE matches
               SET data=?,
               rows=?,
               cols=?   
               WHERE pair_id=?''',
            (array_to_blob(matches),) + matches.shape + (pair_id,))

    def get_matches(self, image_id1, image_id2):
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        rows = self.execute(
            '''SELECT * from matches  
            WHERE pair_id=?''',
            (pair_id,))
        return rows.fetchall()

    def get_only_matches(self, image_id1, image_id2):
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        rows = self.execute(
            '''SELECT * from matches  
            WHERE pair_id=?''',
            (pair_id,))
        rows = blob_to_array(rows.fetchall()[0][3], np.uint32)
        return rows

    def get_all_matches(self):
        rows = self.execute(
            '''SELECT * from matches''')
        return rows.fetchall()

    def delete_desc(self):
        self.execute("DELETE FROM descriptors")

    # .execute("SELECT name FROM sqlite_master WHERE type='table';")

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))

    def get_all_two_view_geometries(self):
        rows = self.execute(
            '''SELECT * from two_view_geometries''')
        return rows.fetchall()
