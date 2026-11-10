def parse_filename(filename):
    
    parts = filename.split('_')
    vehicle_id = int(parts[0])
    camera_id = int(parts[1][1:])  
    
    return (vehicle_id, camera_id)