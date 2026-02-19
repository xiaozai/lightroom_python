# -*- coding: utf-8 -*-
"""Test script - Test color panel, detail panel and import/export functions"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.editor import ImageEditor
import json

def test_color_panel():
    """Test color panel functions"""
    print("=" * 50)
    print("Test Color Panel Functions")
    print("=" * 50)

    editor = ImageEditor()

    # Load test image
    test_image = os.path.expanduser("~/Desktop/test.jpg")
    if not os.path.exists(test_image):
        print(f"Error: Test image not found: {test_image}")
        return False

    editor.load_image(test_image)
    print(f"[OK] Loaded test image: {test_image}")

    # Test white balance
    print("\nTesting white balance...")
    editor.set_param("temp", 30)  # warm tone
    editor.set_param("tint", 20)  # magenta tint
    edited = editor.get_edited()
    print(f"[OK] White balance adjusted - temp: 30, tint: 20")

    # Test HSL
    print("\nTesting HSL color mixer...")
    editor.set_param("hsl_hue_red", 20)
    editor.set_param("hsl_sat_blue", 30)
    editor.set_param("hsl_lum_green", -15)
    edited = editor.get_edited()
    print(f"[OK] HSL adjusted - red hue: 20, blue sat: 30, green lum: -15")

    # Test color grading
    print("\nTesting color grading...")
    editor.set_param("cg_shadows_hue", 230)  # blue shadows
    editor.set_param("cg_shadows_sat", 30)
    editor.set_param("cg_highlights_hue", 45)  # orange highlights
    editor.set_param("cg_highlights_sat", 25)
    editor.set_param("cg_blending", 60)
    editor.set_param("cg_balance", 10)
    edited = editor.get_edited()
    print(f"[OK] Color grading applied")

    return True

def test_detail_panel():
    """Test detail panel functions"""
    print("\n" + "=" * 50)
    print("Test Detail Panel Functions")
    print("=" * 50)

    editor = ImageEditor()

    test_image = os.path.expanduser("~/Desktop/test.jpg")
    editor.load_image(test_image)

    # Test sharpening
    print("\nTesting sharpening...")
    editor.set_param("sharpen_amount", 50)
    editor.set_param("sharpen_radius", 1.5)
    editor.set_param("sharpen_detail", 30)
    editor.set_param("sharpen_masking", 20)
    edited = editor.get_edited()
    print(f"[OK] Sharpening applied - amount: 50, radius: 1.5, detail: 30, masking: 20")

    # Test noise reduction
    print("\nTesting noise reduction...")
    editor.set_param("noise_luminance", 25)
    editor.set_param("noise_color", 30)
    edited = editor.get_edited()
    print(f"[OK] Noise reduction applied - luminance: 25, color: 30")

    # Save test result
    output_path = os.path.expanduser("~/Desktop/test_output.jpg")
    editor.save_image(output_path)
    print(f"[OK] Saved test output: {output_path}")

    return True

def test_import_export():
    """Test import/export functions"""
    print("\n" + "=" * 50)
    print("Test Import/Export Functions")
    print("=" * 50)

    editor = ImageEditor()

    test_image = os.path.expanduser("~/Desktop/test.jpg")
    editor.load_image(test_image)

    # Set some parameters
    print("\nSetting test parameters...")
    editor.set_param("exposure", 15)
    editor.set_param("contrast", 20)
    editor.set_param("temp", -20)
    editor.set_param("sharpen_amount", 40)
    editor.set_param("hsl_sat_blue", 25)

    # Export preset
    preset_path = os.path.expanduser("~/Desktop/test_preset.json")
    params_dict = editor.get_params_dict()
    with open(preset_path, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=2, ensure_ascii=False)
    print(f"[OK] Exported preset: {preset_path}")

    # Reset editor
    editor.reset()
    print("[OK] Reset editor")

    # Verify reset
    assert editor.get_param("exposure") == 0
    assert editor.get_param("contrast") == 0
    print("[OK] Verified parameters reset to 0")

    # Import preset
    with open(preset_path, 'r', encoding='utf-8') as f:
        loaded_params = json.load(f)
    editor.set_params_from_dict(loaded_params)
    print(f"[OK] Imported preset: {preset_path}")

    # Verify import
    assert editor.get_param("exposure") == 15
    assert editor.get_param("contrast") == 20
    assert editor.get_param("temp") == -20
    assert editor.get_param("sharpen_amount") == 40
    assert editor.get_param("hsl_sat_blue") == 25
    print("[OK] Verified imported parameters correct")

    return True

def test_combined_workflow():
    """Test combined workflow"""
    print("\n" + "=" * 50)
    print("Test Combined Workflow")
    print("=" * 50)

    editor = ImageEditor()

    test_image = os.path.expanduser("~/Desktop/test.jpg")
    editor.load_image(test_image)

    # Apply all adjustments
    print("\nApplying combined adjustments...")

    # Basic adjustments
    editor.set_param("exposure", 10)
    editor.set_param("contrast", 15)
    editor.set_param("highlights", -20)
    editor.set_param("shadows", 25)

    # Effect adjustments
    editor.set_param("clarity", 20)
    editor.set_param("dehaze", 15)

    # Color panel
    editor.set_param("temp", -10)
    editor.set_param("vibrance", 15)

    # Detail panel
    editor.set_param("sharpen_amount", 30)
    editor.set_param("noise_luminance", 15)

    # Get edited image
    edited = editor.get_edited()
    print(f"[OK] Combined adjustments completed")

    # Save result
    output_path = os.path.expanduser("~/Desktop/test_combined_output.jpg")
    editor.save_image(output_path)
    print(f"[OK] Saved combined edit result: {output_path}")

    # Export preset
    preset_path = os.path.expanduser("~/Desktop/test_combined_preset.json")
    params_dict = editor.get_params_dict()
    with open(preset_path, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=2, ensure_ascii=False)
    print(f"[OK] Exported combined preset: {preset_path}")

    return True

def main():
    print("\n" + "=" * 60)
    print("  Python Lightroom Tool - Feature Tests")
    print("=" * 60)

    try:
        # Test color panel
        if not test_color_panel():
            print("[FAIL] Color panel test failed")
            return

        # Test detail panel
        if not test_detail_panel():
            print("[FAIL] Detail panel test failed")
            return

        # Test import/export
        if not test_import_export():
            print("[FAIL] Import/export test failed")
            return

        # Test combined workflow
        if not test_combined_workflow():
            print("[FAIL] Combined workflow test failed")
            return

        print("\n" + "=" * 60)
        print("  All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
