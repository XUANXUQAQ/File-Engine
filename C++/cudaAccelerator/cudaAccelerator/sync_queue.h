#pragma once
#include <list>
#include <mutex>
#include <condition_variable>

template<typename T>
class sync_queue
{
	[[nodiscard]] bool is_full() const
	{
		return m_queue_.size() == m_max_size_;
	}

	[[nodiscard]] bool is_empty() const
	{
		return m_queue_.empty();
	}

public:
	explicit sync_queue(const int max_size) : m_max_size_(max_size)
	{
	}

	void put(const T& x)
	{
		std::lock_guard locker(m_mutex_);

		while (is_full())
		{
			m_not_full_.wait(m_mutex_);
		}
		m_queue_.emplace_back(x);
		m_not_empty_.notify_one();
	}

	void take(T& x)
	{
		std::lock_guard locker(m_mutex_);

		while (is_empty())
		{
			m_not_empty_.wait(m_mutex_);
		}
		x = m_queue_.front();
		m_queue_.pop_front();
		m_not_full_.notify_one();
	}

private:
	std::list<T> m_queue_;
	std::mutex m_mutex_;
	std::condition_variable_any m_not_empty_;
	std::condition_variable_any m_not_full_;
	unsigned m_max_size_;
};