import trimesh

def load_mesh(mesh_path,scale=1.0):
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    mesh.apply_scale(scale) #if scale is 1000,the mesh unit will be expanded 1000 times (eg. m to mm
    if not mesh.is_watertight:
        print("Mesh is not watertight, attempting to repair...")
        mesh.fill_holes()
        mesh.merge_vertices()
        if not mesh.is_watertight:
            print("Repair failed, mesh is still not watertight.")
            return None
        else:
            print("Repair successful.")
    print(f"Mesh loaded successfully. Number of vertices: {len(mesh.vertices)}, Number of faces: {len(mesh.faces)}")
    return mesh

def sample_face_points(mesh, num_sample_points=500):
    points = mesh.sample(num_sample_points) # the points are on the faces of the mesh
    return points