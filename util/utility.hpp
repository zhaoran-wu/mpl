//#include <vector>
//#include <igl/ply.h>
//
//
//namespace
//{
//    template <typename Scalar> int ply_type();
//    template <> int ply_type<char>(){ return PLY_CHAR; }
//    template <> int ply_type<short>(){ return PLY_SHORT; }
//    template <> int ply_type<int>(){ return PLY_INT; }
//    template <> int ply_type<unsigned char>(){ return PLY_UCHAR; }
//    template <> int ply_type<unsigned short>(){ return PLY_SHORT; }
//    template <> int ply_type<unsigned int>(){ return PLY_UINT; }
//    template <> int ply_type<float>(){ return PLY_FLOAT; }
//    template <> int ply_type<double>(){ return PLY_DOUBLE; }
//}
//
//template <
//        typename DerivedV,
//        typename DerivedF,
//        typename DerivedUV>
// bool writePLY(
//        const std::string & filename,
//        const Eigen::MatrixBase<DerivedV> & V,
//        const Eigen::MatrixBase<DerivedF> & F,
//        const Eigen::MatrixBase<DerivedUV> & UV_V,
//        const bool ascii)
//{
//    // Largely based on obj2ply.c
//    typedef typename DerivedV::Scalar VScalar;
//    typedef typename DerivedUV::Scalar UVScalar;
//    typedef typename DerivedF::Scalar FScalar;
//
//    typedef struct Vertex
//    {
//        VScalar x,y,z,w;          /* position */
//    } Vertex;
//
//    typedef struct UV
//    {
//        UVScalar s,t;              /* texture coordinates */
//    } UV;
//
//    typedef struct Face
//    {
//        unsigned char nverts;    /* number of vertex indices in list */
//        FScalar *verts;              /* vertex index list */
//    } Face;
//
//    igl::ply::PlyProperty vert_props[] =
//            { /* list of property information for a vertex */
//                    {"x", ply_type<VScalar>(), ply_type<VScalar>(),offsetof(Vertex,x),0,0,0,0},
//                    {"y", ply_type<VScalar>(), ply_type<VScalar>(),offsetof(Vertex,y),0,0,0,0},
//                    {"z", ply_type<VScalar>(), ply_type<VScalar>(),offsetof(Vertex,z),0,0,0,0},
//            };
//
//    igl::ply::PlyProperty uv_vert_props[] =
//            {
//                    {"s", ply_type<UVScalar>(),ply_type<UVScalar>(),offsetof(UV,s),0,0,0,0},
//                    {"t", ply_type<UVScalar>(),ply_type<UVScalar>(),offsetof(UV,t),0,0,0,0},
//            };
//
//    igl::ply::PlyProperty face_props[] =
//            { /* list of property information for a face */
//                    {"vertex_indices", ply_type<FScalar>(), ply_type<FScalar>(),
//                            offsetof(Face,verts), 1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)},
//            };
//    const bool has_texture_coords = UV_V.rows() > 0;
//    std::vector<Vertex> vlist(V.rows());
//    std::vector<Face> flist(F.rows());
//    std::vector<UV> uvlist(UV_V.rows());
//
//    for(size_t i = 0;i<(size_t)V.rows();i++)
//    {
//        vlist[i].x = V(i,0);
//        vlist[i].y = V(i,1);
//        vlist[i].z = V(i,2);
//    }
//
//    for(size_t i = 0;i<(size_t)UV_V.rows();i++)
//    {
//        uvlist[i].s = UV_V(i,0);
//        uvlist[i].t = UV_V(i,1);
//    }
//
//    for(size_t i = 0;i<(size_t)F.rows();i++)
//    {
//        flist[i].nverts = F.cols();
//        flist[i].verts = new FScalar[F.cols()];
//        for(size_t c = 0;c<(size_t)F.cols();c++)
//        {
//            flist[i].verts[c] = F(i,c);
//        }
//    }
//
//    const char * elem_names[] = {"vertex","face","texturecoords"};
//    FILE * fp = fopen(filename.c_str(),"w");
//    if(fp==NULL)
//    {
//        return false;
//    }
//    igl::ply::PlyFile * ply = igl::ply::ply_write(fp, 3,elem_names,
//                                                  (ascii ? PLY_ASCII : PLY_BINARY_LE));
//    if(ply==NULL)
//    {
//        return false;
//    }
//
//    std::vector<igl::ply::PlyProperty> plist;
//    plist.push_back(vert_props[0]);
//    plist.push_back(vert_props[1]);
//    plist.push_back(vert_props[2]);
//    ply_describe_element(ply, "vertex", V.rows(),plist.size(),
//                         &plist[0]);
//
//    ply_describe_element(ply, "face", F.rows(),1,&face_props[0]);
//    ply_describe_element(ply, "texturecoords", UV_V.rows(),2,&uv_vert_props[0]);
//    ply_header_complete(ply);
//    int native_binary_type = igl::ply::get_native_binary_type2();
//    ply_put_element_setup(ply, "vertex");
//    for(const auto v : vlist)
//    {
//        ply_put_element(ply, (void *) &v, &native_binary_type);
//    }
//    ply_put_element_setup(ply, "face");
//    for(const auto f : flist)
//    {
//        ply_put_element(ply, (void *) &f, &native_binary_type);
//    }
//    ply_put_element_setup(ply, "texturecoords");
//    for(const auto u : uvlist)
//    {
//        ply_put_element(ply, (void *) &u, &native_binary_type);
//    }
//
//    ply_close(ply);
//    for(size_t i = 0;i<(size_t)F.rows();i++)
//    {
//        delete[] flist[i].verts;
//    }
//    return true;
//}

namespace util {

template <typename T>
void saveEigenMatrix(const std::string& filename, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << mat;
    } else {
        throw std::runtime_error("could not open " + filename);
    }
    file.close();
}

template <typename T>
void saveCVMatrix(const std::string& filename, const cv::Mat& mat) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int r = 0; r < mat.rows; ++r) {
            for (int c = 0; c < mat.cols; ++c) {
                file << mat.at<T>(r,c) << ' ';
            }
            file << '\n';
        }
    } else {
        throw std::runtime_error("could not open " + filename);
    }
    file.close();
}

} // namespace util