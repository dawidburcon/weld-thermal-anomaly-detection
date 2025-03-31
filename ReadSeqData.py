from flirpy.io.seq import Splitter
import logging

logging.basicConfig(level=logging.DEBUG)

seq_file = "625_38n18_1_2mm_-161_07_41_19_806.seq"
output_folder = "./frames_output"

splitter = Splitter(
    output_folder=output_folder, 
    exiftool_path='/usr/bin/exiftool',
    width=640, 
    height=480
)

splitter.process([seq_file])

print(f"Klatki zapisane w folderze: {output_folder}")






