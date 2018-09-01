/*
Copyright (c) 2016, TU Dresden
Copyright (c) 2017, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#pragma once

#include <lua.hpp>

/**
 * @brief Transfer a matrix to lua.
 * @param mat Input matrix.
 * @param state Lua state.
 */
void pushMat(const cv::Mat_<double>& mat, lua_State* state)
{
    lua_createtable(state, mat.rows * mat.cols, 0);
    int newTable = lua_gettop(state);
  
    int i = 1;
    
    for(int y = 0; y < mat.rows; y++)
    for(int x = 0; x < mat.cols; x++)
    {
	lua_pushnumber(state, mat(y, x));
	lua_rawseti(state, newTable, i++);
    }
}

/**
 * @brief Transfer a list of 3-channel matrices to lua (e.g. for RGB data).
 * @param maps List of 3-channel matrices.
 * @param state Lua state.
 */
void pushMaps(const std::vector<cv::Mat_<cv::Vec3f>>& maps, lua_State* state)
{
    if(maps.empty()) return;
  
    lua_createtable(state, maps.size() * 3 * maps[0].rows * maps[0].cols, 0);
    int newTable = lua_gettop(state);
  
    int i = 1;
    
    for(int n = 0; n < maps.size(); n++)
    for(int c = 0; c < 3; c++)
    for(int y = 0; y < maps[n].rows; y++)
    for(int x = 0; x < maps[n].cols; x++)
    {
	lua_pushnumber(state, maps[n](y, x)[c]);
	lua_rawseti(state, newTable, i++);
    }
}

/**
 * @brief Transfer a list of 1-channel matrices to lua (e.g. for reprojection error images).
 * @param maps List of 3-channel matrices.
 * @param state Lua state.
 */
void pushMaps(const std::vector<cv::Mat_<float>>& maps, lua_State* state)
{
    if(maps.empty()) return;
  
    lua_createtable(state, maps.size() * maps[0].rows * maps[0].cols, 0);
    int newTable = lua_gettop(state);
  
    int i = 1;
    
    for(int n = 0; n < maps.size(); n++)
    for(int y = 0; y < maps[n].rows; y++)
    for(int x = 0; x < maps[n].cols; x++)
    {
	lua_pushnumber(state, maps[n](y, x));
	lua_rawseti(state, newTable, i++);
    }
}

/**
 * @brief Transfer a list of 3D vectors to lua.
 * @param vec List of 3D vectors.
 * @param state Lua state.
 */
void pushVec(const std::vector<cv::Vec3f>& vec, lua_State* state)
{
    if(vec.empty()) return;
  
    lua_createtable(state, vec.size() * 3, 0);
    int newTable = lua_gettop(state);
  
    int i = 1;
    
    for(int n = 0; n < vec.size(); n++)
    {
	lua_pushnumber(state, vec[n][0]);
	lua_rawseti(state, newTable, i++);
	lua_pushnumber(state, vec[n][1]);
	lua_rawseti(state, newTable, i++);
	lua_pushnumber(state, vec[n][2]);
	lua_rawseti(state, newTable, i++);
    }
}

/**
 * @brief Transfer a list of real numbers to lua.
 * @param vec List of real numbers.
 * @param state Lua state.
 */
void pushVec(const std::vector<double>& vec, lua_State* state)
{
    if(vec.empty()) return;
  
    lua_createtable(state, vec.size(), 0);
    int newTable = lua_gettop(state);
  
    for(int n = 0; n < vec.size(); n++)
    {
	lua_pushnumber(state, vec[n]);
	lua_rawseti(state, newTable, n+1);
    }
}

/**
 * @brief Print lua error message.
 * @param Lua state.
 */
void print_error(lua_State* state) 
{
    // The error message is on top of the stack.
    // Fetch it, print it and then pop it off the stack.
    const char* message = lua_tostring(state, -1);
    puts(message);
    lua_pop(state, 1);
}

/**
 * @brief Execute the given lua script.
 * @param filename Lua script.
 * @param state Lua state.
 */
void execute(const char* filename, lua_State* state)
{
    int result;

    result = luaL_loadfile(state, filename);

    if ( result != 0 ) 
    {
	print_error(state);
	return;
    }

    result = lua_pcall(state, 0, LUA_MULTRET, 0);

    if ( result != 0 ) 
    {
	print_error(state);
	return;
    }
}

/**
 * @brief Calls the constructModel function of the given lua state (to construct a CNN).
 *

 * @param inW Width of CNN input.
 * @param inH Height of CNN input.
 * @param outW Width of CNN output.
 * @param outH Height of CNN output.
 * @param state Lua state.
 */
void constructModel(int inW, int inH, int outW, int outH, lua_State* state)
{
    lua_getglobal(state, "constructModel");
    lua_pushnumber(state, inW);
    lua_pushnumber(state, inH);
    lua_pushnumber(state, outW);
    lua_pushnumber(state, outH);
    lua_pcall(state, 4, 0, 0);
}

/**
 * @brief Load a model (CNN) stored in the given file.
 * @param modelFile File storing the model.
 * @param inW Width of CNN input.
 * @param inH Height of CNN input.
 * @param outW Width of CNN output.
 * @param outH Height of CNN output.
 * @param state Lua state.
 */
void loadModel(std::string modelFile, int inW, int inH, int outW, int outH, lua_State* state)
{
    lua_getglobal(state, "loadModel"); 
    lua_pushstring(state, modelFile.c_str());
    lua_pushnumber(state, inW);
    lua_pushnumber(state, inH);
    lua_pushnumber(state, outW);
    lua_pushnumber(state, outH);
    lua_pcall(state, 5, 0, 0);
}

/**
 * @brief Initalize the scoring script.
 * @param threshold Inlier threshold to use in the soft inlier counting.
 * @param outW Width of CNN output.
 * @param outH Height of CNN output.
 * @param state Lua state.
 */
void loadScore(double threshold, int outW, int outH, lua_State* state)
{
    lua_getglobal(state, "loadModel");
    lua_pushnumber(state, threshold);
    lua_pushnumber(state, outW);
    lua_pushnumber(state, outH);
    lua_pcall(state, 3, 0, 0);
}

/**
 * @brief Call the backward function of the given lua state.
 *
 * This function concerns the object coordinate CNN.
 *
 * @param loss Loss measured during the forward pass.
 * @param maps Input data of the forward pass.
 * @param dLoss Gradients of the loss.
 * @param state Lua state.
 */
void backward(
    double loss,
    const std::vector<cv::Mat_<cv::Vec3f>>& maps,
    const cv::Mat_<double>& dLoss,
    lua_State* state)
{
    lua_getglobal(state, "backward"); 
    lua_pushinteger(state, dLoss.rows);
    lua_pushnumber(state, loss);
    pushMaps(maps, state);

    pushMat(dLoss, state);
    lua_pcall(state, 4, 0, 0);
}

/**
 * @brief Call of the forward function of the given lua state.
 *
 * This function concerns the object coordinate CNN.
 *
 * @param maps Input data (eg. RGB images).
 * @param sampling Subsampling map that defines the CNN output size.
 * @return List of predictions.
 */
std::vector<cv::Vec3f> forward(
    const std::vector<cv::Mat_<cv::Vec3f>>& maps,
    const cv::Mat_<cv::Point2i>& sampling,
    lua_State* state)
{
    lua_getglobal(state, "forward"); 
    lua_pushinteger(state, maps.size());
    pushMaps(maps, state);
    lua_pcall(state, 2, 1, 0);

    std::vector<cv::Vec3f> results(sampling.cols * sampling.rows);

    for(unsigned i = 0; i < results.size(); i++)
    for(unsigned j = 0; j < 3; j++)
    {
        int idx = i * 3 + j;
	lua_pushnumber(state, idx + 1);	    
	lua_gettable(state, -2);
	results[i][j] = lua_tonumber(state, -1);
	lua_pop(state, 1);
    }
    lua_pop(state, 1);      

    return results;
}

/**
 * @brief Call of the forward function of the given lua state.
 *
 * This function concerns the score CNN.
 *
 * @param maps Input data (eg. reprojection error images).
 * @param state Lua state.
 * @return List of predictions, one per input datum.
 */
std::vector<double> forward(const std::vector<cv::Mat_<float>>& maps, lua_State* state)
{
    lua_getglobal(state, "forward");
    lua_pushinteger(state, maps.size());
    pushMaps(maps, state);
    lua_pcall(state, 2, maps.size(), 0);

    std::vector<double> results(maps.size());

    for(int i = 0; i < results.size(); i++)
    {
        results[results.size() - 1 - i] = lua_tonumber(state, -1);
        lua_pop(state, 1);
    }

    return results;
}

/**
 * @brief Call the backward function of the given lua state.
 *
 * This function concerns the score CNN.
 *
 * @param maps Input data of the forward pass (eg. reprojection error images).
 * @param state Lua state.
 * @param scoreOutputGradients Gradients at the output of the Score CNN.
 * @param gradients Output parameter. Gradients at the input of the score CNN.
 */
void backward(
    const std::vector<cv::Mat_<float>>& maps,
    lua_State* state,
    const std::vector<double>& scoreOutputGradients,
    std::vector<cv::Mat_<double>>& gradients)
{
    lua_getglobal(state, "backward");
    lua_pushinteger(state, maps.size());
    pushMaps(maps, state);
    pushVec(scoreOutputGradients, state);
    lua_pcall(state, 3, 1, 0);

    gradients.resize(maps.size());

    for(unsigned c = 0; c < maps.size(); c++)
    {
        gradients[c] = cv::Mat_<double>(maps[c].size());

        for(unsigned x = 0; x < gradients[c].cols; x++)
        for(unsigned y = 0; y < gradients[c].rows; y++)
        {
            int idx = c * gradients[c].cols * gradients[c].rows + x * gradients[c].rows + y;
            lua_pushnumber(state, idx + 1);
            lua_gettable(state, -2);
            gradients[c](y, x) = lua_tonumber(state, -1);
            lua_pop(state, 1);
        }
    }
    lua_pop(state, 1);
}

/**
 * @brief Sets CNN model to evaluate mode.
 * @param state LUA state for access to CNN model.
 */
void setEvaluate(lua_State* state)
{
    lua_getglobal(state, "setEvaluate");
    lua_pcall(state, 0, 0, 0);
}

/**
 * @brief Sets CNN model to training mode.
 * @param state LUA state for access to CNN model.
 */
void setTraining(lua_State* state)
{
    lua_getglobal(state, "setTraining");
    lua_pcall(state, 0, 0, 0);
}
