#pragma once
#define NO_COPY(classname)                \
        classname(const classname&) = delete; \
        classname& operator=(const classname&) = delete;
