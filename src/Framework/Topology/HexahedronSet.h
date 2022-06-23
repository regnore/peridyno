#pragma once
#include "QuadSet.h"
namespace dyno
{
	template<typename TDataType>
	class HexahedronSet : public QuadSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Quad Quad;
		typedef typename TopologyModule::Hexahedron Hexahedron;

		HexahedronSet();
		~HexahedronSet();

		//void loadTetFile(std::string filename);

		void setHexahedrons(std::vector<Hexahedron>& hexahedrons);
		void setHexahedrons(DArray<Hexahedron>& hexahedrons);

		DArray<Hexahedron>& getHexahedrons() { return m_hexahedrons; }
		DArray<Tri2Tet>& getQua2Hex() { return quad2Hex; }

		DArrayList<int>& getVer2Hex();

		void getVolume(DArray<Real>& volume);

		void updateQuads();

		void copyFrom(HexahedronSet<TDataType> hexSet);

	protected:
		DArray<Hexahedron> m_hexahedrons;

	private:
		DArray<Quad2Hex> quad2Hex;
		DArrayList<int> m_ver2Hex;
	};
}

