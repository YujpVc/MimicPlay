import pyrealsense2 as rs
import cv2
import numpy as np

def list_devices():
    """List all connected RealSense devices and their serial numbers."""
    print("Searching for RealSense devices...")
    ctx = rs.context()
    devices = ctx.query_devices()
    
    device_list = []
    
    if len(devices) == 0:
        print("No RealSense devices found!")
        return []

    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"[Device {i}] Name: {name}, Serial: {serial}")
        device_list.append(serial)
        
    return device_list

def main():
    serials = list_devices()
    if not serials:
        return

    print(f"\nFound {len(serials)} cameras. Initializing...")

    # Initialize all pipelines
    pipelines = []
    configs = []
    
    try:
        for serial in serials:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start streaming immediately
            print(f"Starting camera {serial}...")
            pipe.start(cfg)
            
            pipelines.append(pipe)
            configs.append(cfg)
            
        print("\nAll cameras started. Press 'q' to exit.")

        while True:
            images_to_show = []
            
            for i, pipe in enumerate(pipelines):
                serial = serials[i]
                
                # Wait for frames (non-blocking if possible, but wait_for_frames is blocking)
                # To prevent one camera blocking others, we use poll_for_frames or wait with short timeout?
                # Usually wait_for_frames is fine if cameras are synced or fast enough.
                # However, for robustness, let's use wait_for_frames with a timeout.
                try:
                    frames = pipe.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    # Timeout or error
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                # Convert to numpy
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Visualize depth
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Add Label
                cv2.putText(color_image, f"Cam {i}: {serial}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Stack for this camera
                cam_stack = np.hstack((color_image, depth_colormap))
                images_to_show.append(cam_stack)
            
            if not images_to_show:
                continue

            # Stack all cameras vertically
            # Resize if needed to fit screen? For 2 cameras it's fine.
            if len(images_to_show) > 0:
                final_image = np.vstack(images_to_show)
                
                cv2.imshow('RealSense Multi-Camera View', final_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {e}")
        
    finally:
        print("Stopping all pipelines...")
        for pipe in pipelines:
            try:
                pipe.stop()
            except:
                pass
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
