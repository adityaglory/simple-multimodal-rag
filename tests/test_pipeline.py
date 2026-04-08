import os
import pytest
from src.vlm import encode_image

def test_encode_image(tmp_path):
    """
    Tests if our image encoder correctly translates binary data to base64.
    We use pytest's built-in 'tmp_path' to create a fake image safely.
    """
    # Create a temporary file
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"fake image data")

    # Run our function
    base64_str = encode_image(str(fake_image))
    
    # Assertions check if the code behaved as expected
    assert isinstance(base64_str, str), "Image was not converted to a string"
    assert len(base64_str) > 0, "The encoded string is empty"

def test_missing_image():
    """Tests if our code properly handles a missing file error."""
    with pytest.raises(FileNotFoundError):
        encode_image("non_existent_image.jpg")

def test_project_structure():
    """Ensures critical folders exist so the Docker container doesn't crash."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    assert os.path.exists(os.path.join(base_dir, "data")), "Data folder is missing!"
    assert os.path.exists(os.path.join(base_dir, "src")), "Source folder is missing!"