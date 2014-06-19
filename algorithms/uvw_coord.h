#pragma once
namespace imaging{
	template<typename T>
	struct uvw_coord{
		T _u,_v,_w;
		uvw_coord(T u = 0, T v = 0, T w = 0):_u(u),_v(v),_w(w){}
	};
}
