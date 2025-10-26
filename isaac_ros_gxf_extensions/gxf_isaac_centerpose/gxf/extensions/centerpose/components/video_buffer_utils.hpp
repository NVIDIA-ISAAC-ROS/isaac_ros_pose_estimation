// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "gxf/multimedia/video.hpp"
#include "gxf/std/allocator.hpp"

namespace nvidia {
namespace isaac {

template <gxf::VideoFormat T>
inline constexpr uint8_t GetChannelSize();

template <>
inline constexpr uint8_t GetChannelSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB>() {
  return 3;
}

template <>
inline constexpr uint8_t GetChannelSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR>() {
  return 3;
}

template <>
inline constexpr uint8_t GetChannelSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY>() {
  return 1;
}

template <gxf::VideoFormat T>
inline const char* GetColorName();

template <>
inline const char* GetColorName<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB>() {
  return "RGB";
}

template <>
inline const char* GetColorName<gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR>() {
  return "BGR";
}

template <>
inline const char* GetColorName<gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY>() {
  return "gray";
}

template <gxf::VideoFormat T>
gxf::Expected<void> AllocateVideoBuffer(
    gxf::Handle<gxf::VideoBuffer> video_buffer, size_t width, size_t height,
    gxf::MemoryStorageType memory_storage_type, gxf::Handle<gxf::Allocator> allocator) {
  if (width % 2 != 0 || height % 2 != 0) {
    GXF_LOG_ERROR(
        "Image width and height must be even, but received width: %zu, height: %zu", width, height);
    return gxf::Unexpected{GXF_FAILURE};
  }

  std::array<gxf::ColorPlane, 1> planes{
      gxf::ColorPlane(GetColorName<T>(), GetChannelSize<T>(), GetChannelSize<T>() * width)};
  gxf::VideoFormatSize<T> video_format_size;
  const uint64_t size = video_format_size.size(width, height, planes);
  const std::vector<gxf::ColorPlane> planes_filled{planes.begin(), planes.end()};
  const gxf::VideoBufferInfo buffer_info{
      static_cast<uint32_t>(width), static_cast<uint32_t>(height), T, planes_filled,
      gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
  return video_buffer->resizeCustom(buffer_info, size, memory_storage_type, allocator);
}

}  // namespace isaac
}  // namespace nvidia
