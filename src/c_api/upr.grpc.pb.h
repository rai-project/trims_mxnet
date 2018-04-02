// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: upr.proto
#ifndef GRPC_upr_2eproto__INCLUDED
#define GRPC_upr_2eproto__INCLUDED

#include "upr.pb.h"

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/method_handler_impl.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

namespace grpc {
class CompletionQueue;
class Channel;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace upr {

class Registry final {
 public:
  static constexpr char const* service_full_name() {
    return "upr.Registry";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status Open(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::upr::ModelHandle* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::ModelHandle>> AsyncOpen(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::ModelHandle>>(AsyncOpenRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::ModelHandle>> PrepareAsyncOpen(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::ModelHandle>>(PrepareAsyncOpenRaw(context, request, cq));
    }
    virtual ::grpc::Status Close(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::upr::Void* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::Void>> AsyncClose(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::Void>>(AsyncCloseRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::Void>> PrepareAsyncClose(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::Void>>(PrepareAsyncCloseRaw(context, request, cq));
    }
    virtual ::grpc::Status Info(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::upr::Model* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::Model>> AsyncInfo(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::Model>>(AsyncInfoRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::Model>> PrepareAsyncInfo(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::upr::Model>>(PrepareAsyncInfoRaw(context, request, cq));
    }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::upr::ModelHandle>* AsyncOpenRaw(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::upr::ModelHandle>* PrepareAsyncOpenRaw(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::upr::Void>* AsyncCloseRaw(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::upr::Void>* PrepareAsyncCloseRaw(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::upr::Model>* AsyncInfoRaw(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::upr::Model>* PrepareAsyncInfoRaw(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status Open(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::upr::ModelHandle* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::ModelHandle>> AsyncOpen(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::ModelHandle>>(AsyncOpenRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::ModelHandle>> PrepareAsyncOpen(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::ModelHandle>>(PrepareAsyncOpenRaw(context, request, cq));
    }
    ::grpc::Status Close(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::upr::Void* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::Void>> AsyncClose(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::Void>>(AsyncCloseRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::Void>> PrepareAsyncClose(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::Void>>(PrepareAsyncCloseRaw(context, request, cq));
    }
    ::grpc::Status Info(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::upr::Model* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::Model>> AsyncInfo(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::Model>>(AsyncInfoRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::Model>> PrepareAsyncInfo(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::upr::Model>>(PrepareAsyncInfoRaw(context, request, cq));
    }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader< ::upr::ModelHandle>* AsyncOpenRaw(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::upr::ModelHandle>* PrepareAsyncOpenRaw(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::upr::Void>* AsyncCloseRaw(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::upr::Void>* PrepareAsyncCloseRaw(::grpc::ClientContext* context, const ::upr::ModelHandle& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::upr::Model>* AsyncInfoRaw(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::upr::Model>* PrepareAsyncInfoRaw(::grpc::ClientContext* context, const ::upr::ModelRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_Open_;
    const ::grpc::internal::RpcMethod rpcmethod_Close_;
    const ::grpc::internal::RpcMethod rpcmethod_Info_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status Open(::grpc::ServerContext* context, const ::upr::ModelRequest* request, ::upr::ModelHandle* response);
    virtual ::grpc::Status Close(::grpc::ServerContext* context, const ::upr::ModelHandle* request, ::upr::Void* response);
    virtual ::grpc::Status Info(::grpc::ServerContext* context, const ::upr::ModelRequest* request, ::upr::Model* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_Open : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_Open() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_Open() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Open(::grpc::ServerContext* context, const ::upr::ModelRequest* request, ::upr::ModelHandle* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestOpen(::grpc::ServerContext* context, ::upr::ModelRequest* request, ::grpc::ServerAsyncResponseWriter< ::upr::ModelHandle>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_Close : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_Close() {
      ::grpc::Service::MarkMethodAsync(1);
    }
    ~WithAsyncMethod_Close() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Close(::grpc::ServerContext* context, const ::upr::ModelHandle* request, ::upr::Void* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestClose(::grpc::ServerContext* context, ::upr::ModelHandle* request, ::grpc::ServerAsyncResponseWriter< ::upr::Void>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_Info : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_Info() {
      ::grpc::Service::MarkMethodAsync(2);
    }
    ~WithAsyncMethod_Info() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Info(::grpc::ServerContext* context, const ::upr::ModelRequest* request, ::upr::Model* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestInfo(::grpc::ServerContext* context, ::upr::ModelRequest* request, ::grpc::ServerAsyncResponseWriter< ::upr::Model>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_Open<WithAsyncMethod_Close<WithAsyncMethod_Info<Service > > > AsyncService;
  template <class BaseClass>
  class WithGenericMethod_Open : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_Open() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_Open() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Open(::grpc::ServerContext* context, const ::upr::ModelRequest* request, ::upr::ModelHandle* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_Close : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_Close() {
      ::grpc::Service::MarkMethodGeneric(1);
    }
    ~WithGenericMethod_Close() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Close(::grpc::ServerContext* context, const ::upr::ModelHandle* request, ::upr::Void* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_Info : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_Info() {
      ::grpc::Service::MarkMethodGeneric(2);
    }
    ~WithGenericMethod_Info() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Info(::grpc::ServerContext* context, const ::upr::ModelRequest* request, ::upr::Model* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Open : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_Open() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler< ::upr::ModelRequest, ::upr::ModelHandle>(std::bind(&WithStreamedUnaryMethod_Open<BaseClass>::StreamedOpen, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Open() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Open(::grpc::ServerContext* context, const ::upr::ModelRequest* request, ::upr::ModelHandle* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedOpen(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::upr::ModelRequest,::upr::ModelHandle>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Close : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_Close() {
      ::grpc::Service::MarkMethodStreamed(1,
        new ::grpc::internal::StreamedUnaryHandler< ::upr::ModelHandle, ::upr::Void>(std::bind(&WithStreamedUnaryMethod_Close<BaseClass>::StreamedClose, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Close() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Close(::grpc::ServerContext* context, const ::upr::ModelHandle* request, ::upr::Void* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedClose(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::upr::ModelHandle,::upr::Void>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Info : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_Info() {
      ::grpc::Service::MarkMethodStreamed(2,
        new ::grpc::internal::StreamedUnaryHandler< ::upr::ModelRequest, ::upr::Model>(std::bind(&WithStreamedUnaryMethod_Info<BaseClass>::StreamedInfo, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_Info() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Info(::grpc::ServerContext* context, const ::upr::ModelRequest* request, ::upr::Model* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedInfo(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::upr::ModelRequest,::upr::Model>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_Open<WithStreamedUnaryMethod_Close<WithStreamedUnaryMethod_Info<Service > > > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_Open<WithStreamedUnaryMethod_Close<WithStreamedUnaryMethod_Info<Service > > > StreamedService;
};

}  // namespace upr


#endif  // GRPC_upr_2eproto__INCLUDED