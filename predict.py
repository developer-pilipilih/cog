from cog import BasePredictor, Path, Input
import os
import sys
import shutil
import subprocess
from typing import List

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        # Ensure models are downloaded during setup
        subprocess.run(["python", "facefusion.py", "force-download"], check=True)
    
    def predict(self,
                source_image: Path = Input(description="Source face image"),
                target_image: Path = Input(description="Target image to apply the face to"),
                face_recognizer: str = Input(description="Face recognition model", default="insightface", choices=["insightface"]),
                face_analyser: str = Input(description="Face analysis model", default="insightface", choices=["insightface"]),
                face_detector: str = Input(description="Face detector to use", default="yolov8n", choices=["yolov8n", "retinaface"]),
                face_enhancer: str = Input(description="Face enhancer to use", default="gfpgan", choices=["gfpgan", "codeformer", "restoreformer"]),
                face_enhancer_blend: float = Input(description="Blend factor for face enhancement", default=0.5, ge=0, le=1)
               ) -> Path:
        """Run face fusion prediction"""
        
        # Create temp directories
        os.makedirs("temp_input", exist_ok=True)
        os.makedirs("temp_output", exist_ok=True)
        
        # Copy input files
        source_path = os.path.join("temp_input", "source.jpg")
        target_path = os.path.join("temp_input", "target.jpg")
        output_path = os.path.join("temp_output", "result.jpg")
        
        shutil.copy(str(source_image), source_path)
        shutil.copy(str(target_image), target_path)
        
        # Build command
        cmd = [
            "python", "facefusion.py",
            "headless-run",
            "--source", source_path,
            "--target", target_path,
            "--output", output_path,
            "--face-recognizer", face_recognizer,
            "--face-analyser", face_analyser,
            "--face-detector", face_detector
        ]
        
        # Add enhancer if specified
        if face_enhancer:
            cmd.extend(["--face-enhancer", face_enhancer])
            cmd.extend(["--face-enhancer-blend", str(face_enhancer_blend)])
        
        # Run FaceFusion
        subprocess.run(cmd, check=True)
        
        # Return result if it exists
        if os.path.exists(output_path):
            return Path(output_path)
        else:
            raise RuntimeError("FaceFusion failed to generate an output")
