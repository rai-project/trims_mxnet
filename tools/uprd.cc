#include <iostream>
#include "mxnet/c_api.h"
#include "mxnet/c_predict_api.h"
#include "ipc.h"
#include "upr.pb.h"

using namespace upr;

int main() {
    int version = 0;
    const auto err = MXGetVersion(&version);
    if (err) {
        std::cerr << "error :: " << err << " while getting mxnet version\n";
    }
    std::cout << "in upd. using mxnet version = " << version << " on address  = " << server::address << "\n";
    return 0;
}