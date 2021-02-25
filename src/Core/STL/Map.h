#ifndef MAP_H
#define MAP_H

#include "Platform.h"
#include "Pair.h"

namespace dyno
{
	/**
	 * @brief Be aware do not use this structure on GPU if the data size is large.
	 * 
	 * @tparam T 
	 */
	template <typename MKey, typename T>
	class Map
	{
	public:
		using iterator = Pair<MKey, T>*;

		DYN_FUNC Map();

		DYN_FUNC void reserve(Pair<MKey, T>* buf, int maxSize) {
			m_pairs = buf;
			m_maxSize = maxSize;
		}
		
		DYN_FUNC iterator find(MKey key);

		DYN_FUNC inline iterator begin() {
			return m_pairs;
		};

		DYN_FUNC inline iterator end() {
			return m_pairs + m_size;
		}

		DYN_FUNC void clear();

		DYN_FUNC int size();

		DYN_FUNC iterator insert(Pair<MKey, T> pair);
		DYN_FUNC bool empty();

		DYN_FUNC int erase(const T val);
		DYN_FUNC void erase(iterator val_ptr);

	private:
		int m_size = 0;

		Pair<MKey, T>* m_pairs = nullptr;
		int m_maxSize = 0;
	};
}

#include "Map.inl"

#endif // MAP_H
