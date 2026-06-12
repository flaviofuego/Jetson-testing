#pragma once
// Minimal PLY I/O replacing the Open3D C++ dependency in LightweightMR's
// Delaunay tools (create_delaunay / create_mesh).
//
// Why: Open3D's prebuilt devel SDK is compiled with clang/libc++ (std::__1
// symbols) and cannot be linked from a libstdc++/gcc toolchain alongside the
// system boost. The tools only need "read PLY vertices" and "write PLY mesh",
// so this header provides exactly that.
//
// Supported input:  ascii 1.0 and binary_little_endian 1.0, with the "vertex"
// element first (true for trimesh and Open3D exports). Extra vertex
// properties (colors, normals) are skipped.
// Output: binary_little_endian PLY with float vertices and uchar/int32 faces.

#include <Eigen/Core>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace miniply {

inline size_t TypeSize(const std::string& t) {
    if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
    if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
    if (t == "int" || t == "uint" || t == "int32" || t == "uint32" ||
        t == "float" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    throw std::runtime_error("miniply: unknown PLY type: " + t);
}

inline double ReadScalar(const char* p, const std::string& t) {
    if (t == "float" || t == "float32")  { float v;    std::memcpy(&v, p, 4); return v; }
    if (t == "double" || t == "float64") { double v;   std::memcpy(&v, p, 8); return v; }
    if (t == "int" || t == "int32")      { int32_t v;  std::memcpy(&v, p, 4); return v; }
    if (t == "uint" || t == "uint32")    { uint32_t v; std::memcpy(&v, p, 4); return v; }
    if (t == "short" || t == "int16")    { int16_t v;  std::memcpy(&v, p, 2); return v; }
    if (t == "ushort" || t == "uint16")  { uint16_t v; std::memcpy(&v, p, 2); return v; }
    if (t == "char" || t == "int8")      { int8_t v;   std::memcpy(&v, p, 1); return v; }
    if (t == "uchar" || t == "uint8")    { uint8_t v;  std::memcpy(&v, p, 1); return v; }
    throw std::runtime_error("miniply: unknown PLY type: " + t);
}

inline std::vector<Eigen::Vector3d> ReadPlyPoints(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("miniply: cannot open " + path);

    struct Prop { std::string type, name; };
    std::string line, format;
    size_t n_vertices = 0;
    std::vector<Prop> vprops;
    bool in_vertex_elem = false, seen_vertex_elem = false;

    std::getline(in, line);
    if (line.rfind("ply", 0) != 0)
        throw std::runtime_error("miniply: not a PLY file: " + path);

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string tok;
        ls >> tok;
        if (tok == "format") {
            ls >> format;
        } else if (tok == "comment") {
            continue;
        } else if (tok == "element") {
            std::string name; size_t cnt;
            ls >> name >> cnt;
            if (name == "vertex") {
                if (seen_vertex_elem)
                    throw std::runtime_error("miniply: duplicate vertex element");
                n_vertices = cnt;
                in_vertex_elem = seen_vertex_elem = true;
            } else {
                if (!seen_vertex_elem)
                    throw std::runtime_error(
                        "miniply: vertex element must come first in " + path);
                in_vertex_elem = false;  // faces etc. after vertices — ignored
            }
        } else if (tok == "property") {
            if (!in_vertex_elem) continue;
            Prop p; std::string t;
            ls >> t;
            if (t == "list")
                throw std::runtime_error("miniply: list property in vertex element");
            p.type = t;
            ls >> p.name;
            vprops.push_back(p);
        } else if (tok == "end_header") {
            break;
        }
    }
    if (format != "ascii" && format != "binary_little_endian")
        throw std::runtime_error("miniply: unsupported PLY format: " + format);
    if (n_vertices == 0)
        throw std::runtime_error("miniply: no vertices in " + path);

    int ix = -1, iy = -1, iz = -1;
    size_t row_size = 0;
    std::vector<size_t> offsets(vprops.size());
    for (size_t i = 0; i < vprops.size(); ++i) {
        offsets[i] = row_size;
        row_size += TypeSize(vprops[i].type);
        if (vprops[i].name == "x") ix = static_cast<int>(i);
        if (vprops[i].name == "y") iy = static_cast<int>(i);
        if (vprops[i].name == "z") iz = static_cast<int>(i);
    }
    if (ix < 0 || iy < 0 || iz < 0)
        throw std::runtime_error("miniply: missing x/y/z properties in " + path);

    std::vector<Eigen::Vector3d> points(n_vertices);
    if (format == "binary_little_endian") {
        std::vector<char> row(row_size);
        for (size_t i = 0; i < n_vertices; ++i) {
            in.read(row.data(), static_cast<std::streamsize>(row_size));
            if (!in) throw std::runtime_error("miniply: truncated PLY: " + path);
            points[i] = Eigen::Vector3d(
                ReadScalar(row.data() + offsets[ix], vprops[ix].type),
                ReadScalar(row.data() + offsets[iy], vprops[iy].type),
                ReadScalar(row.data() + offsets[iz], vprops[iz].type));
        }
    } else {  // ascii
        std::vector<double> vals(vprops.size());
        for (size_t i = 0; i < n_vertices; ++i) {
            for (size_t j = 0; j < vprops.size(); ++j)
                if (!(in >> vals[j]))
                    throw std::runtime_error("miniply: truncated PLY: " + path);
            points[i] = Eigen::Vector3d(vals[ix], vals[iy], vals[iz]);
        }
    }
    return points;
}

inline void WritePlyMesh(const std::string& path,
                         const std::vector<Eigen::Vector3d>& vertices,
                         const std::vector<Eigen::Vector3i>& triangles) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("miniply: cannot write " + path);
    out << "ply\nformat binary_little_endian 1.0\n"
        << "element vertex " << vertices.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "element face " << triangles.size() << "\n"
        << "property list uchar int vertex_indices\nend_header\n";
    for (const auto& v : vertices) {
        float xyz[3] = {static_cast<float>(v.x()),
                        static_cast<float>(v.y()),
                        static_cast<float>(v.z())};
        out.write(reinterpret_cast<const char*>(xyz), 12);
    }
    for (const auto& t : triangles) {
        uint8_t n = 3;
        int32_t idx[3] = {t.x(), t.y(), t.z()};
        out.write(reinterpret_cast<const char*>(&n), 1);
        out.write(reinterpret_cast<const char*>(idx), 12);
    }
}

}  // namespace miniply
