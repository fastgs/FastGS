# import open3d as o3d

# # 读取ply文件
# mesh = o3d.io.read_triangle_mesh("/wtc/ssd/workingspace/enhance/PGSR+ours/output_tnt/Truck/mesh/tsdf_fusion_post.ply")

# # 如果是点云文件，可以用 read_point_cloud
# # pcd = o3d.io.read_point_cloud("your_file.ply")

# # 打印一下信息
# print(mesh)

# # 可视化
# o3d.visualization.draw_geometries([mesh])

# import open3d as o3d

# mesh = o3d.io.read_triangle_mesh("/wtc/ssd/workingspace/enhance/PGSR+ours/output_tnt/Truck/mesh/tsdf_fusion_post.ply")

# # 清除顶点颜色（如果有的话）
# mesh.vertex_colors = o3d.utility.Vector3dVector()

# o3d.visualization.draw_geometries([mesh])

import open3d as o3d

# 读取 mesh
mesh = o3d.io.read_triangle_mesh("/wtc/ssd/workingspace/enhance/PGSR+ours/output_tnt/Truck/mesh/tsdf_fusion.ply")

# 计算法线
mesh.compute_vertex_normals()

# 清除颜色信息（避免 ply 里自带颜色覆盖光照效果）
mesh.vertex_colors = o3d.utility.Vector3dVector()

# 可视化：带光照，法线效果明显
o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_back_face=True,   # 显示背面
    mesh_show_wireframe=False   # 关闭线框（可选）
)

